"""KeepGPU local service.

Supports JSON-RPC over stdio/HTTP and REST-style HTTP endpoints.

JSON-RPC methods:
  - start_keep(gpu_ids, vram, interval, busy_threshold, job_id)
  - stop_keep(job_id=None)  # None stops all
  - status(job_id=None)     # None lists all
  - list_gpus()

HTTP endpoints:
  - GET /health
  - GET /api/gpus
  - GET /api/sessions
  - GET /api/sessions/{job_id}
  - POST /api/sessions
  - DELETE /api/sessions
  - DELETE /api/sessions/{job_id}
  - GET / (dashboard static assets)
"""

from __future__ import annotations

import atexit
import json
import sys
import uuid
import argparse
import threading
import mimetypes
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import unquote, urlparse

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities.gpu_info import get_gpu_info
from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_JSON_BODY_BYTES = 1_000_000


@dataclass
class Session:
    controller: GlobalGPUController
    params: Dict[str, Any]


class KeepGPUServer:
    def __init__(
        self,
        controller_factory: Optional[Callable[..., GlobalGPUController]] = None,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._sessions_lock = threading.RLock()
        self._controller_factory = controller_factory or GlobalGPUController
        atexit.register(self.shutdown)

    @staticmethod
    def _release_with_timeout(
        controller: GlobalGPUController,
        timeout_s: float = 10.0,
    ) -> bool:
        done = threading.Event()
        error_holder: Dict[str, Exception] = {}

        def _release() -> None:
            try:
                controller.release()
            except Exception as exc:  # pragma: no cover - defensive
                error_holder["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(target=_release, daemon=True)
        thread.start()
        if not done.wait(timeout_s):
            return False
        if "error" in error_holder:
            raise error_holder["error"]
        return True

    def start_keep(
        self,
        gpu_ids: Optional[List[int]] = None,
        vram: str = "1GiB",
        interval: int = 300,
        busy_threshold: int = -1,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a KeepGPU session that reserves VRAM on one or more GPUs.

        Args:
            gpu_ids: GPU indices to target; None uses all available GPUs.
            vram: Human-readable VRAM size to keep (for example, "1GiB").
            interval: Seconds between controller checks/actions.
            busy_threshold: Utilization above which the controller backs off.
            job_id: Optional session identifier; a UUID is generated if omitted.

        Returns:
            Dict with the started session's job_id, e.g. ``{"job_id": "<id>"}``.

        Raises:
            ValueError: If the provided job_id already exists.
        """
        job_id = job_id or str(uuid.uuid4())
        with self._sessions_lock:
            if job_id in self._sessions:
                raise ValueError(f"job_id {job_id} already exists")

        controller = self._controller_factory(
            gpu_ids=gpu_ids,
            interval=interval,
            vram_to_keep=vram,
            busy_threshold=busy_threshold,
        )
        controller.keep()
        with self._sessions_lock:
            if job_id in self._sessions:
                controller.release()
                raise ValueError(f"job_id {job_id} already exists")
            self._sessions[job_id] = Session(
                controller=controller,
                params={
                    "gpu_ids": gpu_ids,
                    "vram": vram,
                    "interval": interval,
                    "busy_threshold": busy_threshold,
                },
            )
        logger.info("Started keep session %s on GPUs %s", job_id, gpu_ids)
        return {"job_id": job_id}

    def stop_keep(
        self, job_id: Optional[str] = None, quiet: bool = False
    ) -> Dict[str, Any]:
        """
        Stop one or all active keep sessions.

        If job_id is supplied, only that session is stopped; otherwise all active
        sessions are released. When quiet=True, informational logging is skipped.

        Args:
            job_id: Session identifier to stop; None stops every session.
            quiet: Suppress informational logs about stopped sessions.

        Returns:
            Dict with a "stopped" list of job ids. If a specific job_id was not
            found, a "message" field explains the miss.
        """
        if job_id:
            with self._sessions_lock:
                session = self._sessions.pop(job_id, None)
            if session:
                released = self._release_with_timeout(session.controller)
                if not quiet:
                    if released:
                        logger.info("Stopped keep session %s", job_id)
                    else:
                        logger.warning(
                            "Timed out while stopping keep session %s", job_id
                        )
                if released:
                    return {"stopped": [job_id]}
                return {
                    "stopped": [],
                    "timed_out": [job_id],
                    "message": "Timed out while stopping session; release continues in background.",
                }
            return {"stopped": [], "message": "job_id not found"}

        with self._sessions_lock:
            session_items = list(self._sessions.items())
            self._sessions.clear()
        stopped_ids = [jid for jid, _ in session_items]
        timed_out_ids: List[str] = []
        for job_id, session in session_items:
            released = self._release_with_timeout(session.controller)
            if not released:
                timed_out_ids.append(job_id)
        successful = [jid for jid in stopped_ids if jid not in timed_out_ids]
        if successful and not quiet:
            logger.info("Stopped sessions: %s", successful)
        if timed_out_ids and not quiet:
            logger.warning("Timed out stopping sessions: %s", timed_out_ids)
        result: Dict[str, Any] = {"stopped": successful}
        if timed_out_ids:
            result["timed_out"] = timed_out_ids
            result["message"] = (
                "Some sessions timed out during stop; release continues in background."
            )
        return result

    def status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        if job_id:
            with self._sessions_lock:
                session = self._sessions.get(job_id)
            if not session:
                return {"active": False, "job_id": job_id}
            return {
                "active": True,
                "job_id": job_id,
                "params": session.params,
            }
        with self._sessions_lock:
            session_items = list(self._sessions.items())
        return {
            "active_jobs": [
                {"job_id": jid, "params": sess.params} for jid, sess in session_items
            ]
        }

    def list_gpus(self) -> Dict[str, Any]:
        """Return detailed GPU info (id, name, memory, utilization)."""
        infos = get_gpu_info()
        return {"gpus": infos}

    def shutdown(self) -> None:
        """Stop all sessions quietly; ignore errors during interpreter teardown."""
        try:
            self.stop_keep(None, quiet=True)
        except Exception:  # pragma: no cover - defensive
            # Avoid noisy errors during interpreter teardown
            return


def _handle_request(server: KeepGPUServer, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a JSON-RPC payload to the server and return a response dict.

    Args:
        server: Target KeepGPUServer.
        payload: Dict with "method", optional "params", and optional "id".

    Returns:
        JSON-RPC-style dict containing either "result" or "error" plus "id".
    """
    method = payload.get("method")
    params = payload.get("params", {}) or {}
    req_id = payload.get("id")
    try:
        if method == "start_keep":
            result = server.start_keep(**params)
        elif method == "stop_keep":
            result = server.stop_keep(**params)
        elif method == "status":
            result = server.status(**params)
        elif method == "list_gpus":
            result = server.list_gpus()
        else:
            raise ValueError(f"Unknown method: {method}")
        return {"id": req_id, "result": result}
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Request failed")
        return {"id": req_id, "error": {"message": str(exc)}}


class _JSONRPCHandler(BaseHTTPRequestHandler):
    server_version = "KeepGPU-MCP/0.1"

    def _json_response(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError(
                f"Request body too large: {length} bytes (max {MAX_JSON_BODY_BYTES})"
            )
        body = self.rfile.read(length).decode("utf-8")
        return json.loads(body)

    def _serve_static(self, request_path: str) -> None:
        if request_path in ("/", ""):
            relative = "index.html"
        else:
            relative = request_path.lstrip("/")

        requested = (STATIC_DIR / unquote(relative)).resolve()
        static_root = STATIC_DIR.resolve()
        if static_root not in requested.parents and requested != static_root:
            self._json_response(403, {"error": {"message": "Forbidden"}})
            return

        if not requested.exists() or requested.is_dir():
            # SPA fallback for client-side routes.
            requested = static_root / "index.html"
            if not requested.exists():
                self._json_response(404, {"error": {"message": "UI not built"}})
                return

        content = requested.read_bytes()
        content_type, _ = mimetypes.guess_type(str(requested))
        self.send_response(200)
        self.send_header("content-type", content_type or "application/octet-stream")
        self.send_header("content-length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        server_ref = self.server.keepgpu_server  # type: ignore[attr-defined]

        if path == "/health":
            self._json_response(200, {"ok": True})
            return
        if path == "/api/gpus":
            self._json_response(200, server_ref.list_gpus())
            return
        if path == "/api/sessions":
            self._json_response(200, server_ref.status())
            return
        if path.startswith("/api/sessions/"):
            job_id = unquote(path.rsplit("/", 1)[-1]).strip()
            if not job_id:
                self._json_response(400, {"error": {"message": "Missing job_id"}})
                return
            self._json_response(200, server_ref.status(job_id=job_id))
            return

        if path.startswith("/api/"):
            self._json_response(404, {"error": {"message": "Unknown endpoint"}})
            return

        self._serve_static(path)

    def do_POST(self):  # noqa: N802
        """
        Handle an HTTP JSON-RPC request and write a JSON response.

        Expects application/json bodies containing {"method", "params", "id"}.
        Returns 400 with an error object if parsing fails.
        """
        parsed = urlparse(self.path)
        path = parsed.path
        server_ref = self.server.keepgpu_server  # type: ignore[attr-defined]
        try:
            payload = self._read_json_body()
            if path == "/api/sessions":
                allowed_fields = {
                    "gpu_ids",
                    "vram",
                    "interval",
                    "busy_threshold",
                    "job_id",
                }
                unknown_fields = set(payload) - allowed_fields
                if unknown_fields:
                    raise ValueError(
                        f"Unknown request fields: {sorted(unknown_fields)}"
                    )

                safe_payload = {
                    key: value
                    for key, value in payload.items()
                    if key in allowed_fields
                }
                gpu_ids = safe_payload.get("gpu_ids")
                if gpu_ids is not None:
                    if not isinstance(gpu_ids, list):
                        raise ValueError("gpu_ids must be a list of integers")
                    if len(gpu_ids) > 64:
                        raise ValueError("gpu_ids has too many items")
                    if any(
                        (not isinstance(gpu_id, int) or gpu_id < 0)
                        for gpu_id in gpu_ids
                    ):
                        raise ValueError("gpu_ids must contain non-negative integers")
                    visible_gpus = server_ref.list_gpus().get("gpus", [])
                    if visible_gpus and any(
                        gpu_id >= len(visible_gpus) for gpu_id in gpu_ids
                    ):
                        raise ValueError(
                            f"gpu_ids must be in range 0..{len(visible_gpus) - 1}"
                        )

                result = server_ref.start_keep(**safe_payload)
                self._json_response(200, result)
                return

            # JSON-RPC compatibility endpoint.
            if path in ("/", "/rpc"):
                response = _handle_request(server_ref, payload)
                self._json_response(200, response)
                return

            self._json_response(404, {"error": {"message": "Unknown endpoint"}})
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError, TypeError) as exc:
            self._json_response(400, {"error": {"message": f"Bad request: {exc}"}})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("POST request failed for path %s", path)
            self._json_response(
                500,
                {
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                    }
                },
            )

    def do_DELETE(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        server_ref = self.server.keepgpu_server  # type: ignore[attr-defined]

        if path == "/api/sessions":
            self._json_response(200, server_ref.stop_keep(job_id=None))
            return
        if path.startswith("/api/sessions/"):
            job_id = unquote(path.rsplit("/", 1)[-1]).strip()
            if not job_id:
                self._json_response(400, {"error": {"message": "Missing job_id"}})
                return
            self._json_response(200, server_ref.stop_keep(job_id=job_id))
            return
        self._json_response(404, {"error": {"message": "Unknown endpoint"}})

    def log_message(self, format, *args):  # noqa: A003
        """Suppress default request logging."""
        return


def run_stdio(server: KeepGPUServer) -> None:
    """Serve JSON-RPC requests over stdin/stdout (one JSON object per line)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            response = _handle_request(server, payload)
        except Exception as exc:
            response = {"error": {"message": str(exc)}}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


def run_http(server: KeepGPUServer, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run a lightweight HTTP JSON-RPC server on the given host/port."""

    class _Server(ThreadingMixIn, TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = _Server((host, port), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]

    def _serve():
        """Run the HTTP server loop until shutdown."""
        httpd.serve_forever()

    thread = threading.Thread(target=_serve)
    thread.start()
    logger.info(
        "MCP HTTP server listening on http://%s:%s", host, httpd.server_address[1]
    )
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()


def main() -> None:
    """CLI entry point for the KeepGPU MCP server."""
    parser = argparse.ArgumentParser(description="KeepGPU MCP server")
    parser.add_argument(
        "--mode",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (http mode)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (http mode)")
    args = parser.parse_args()

    server = KeepGPUServer()
    if args.mode == "stdio":
        run_stdio(server)
    else:
        run_http(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
