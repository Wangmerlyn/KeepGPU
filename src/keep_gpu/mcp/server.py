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
from keep_gpu.utilities.humanized_input import parse_vram_to_elements
from keep_gpu.utilities.gpu_info import get_gpu_info
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.session_config import (
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
    validate_job_id,
)

logger = setup_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_JSON_BODY_BYTES = 1_000_000


@dataclass
class Session:
    controller: GlobalGPUController
    params: Dict[str, Any]
    state: str = "active"
    last_error: Optional[str] = None


class KeepGPUServer:
    def __init__(
        self,
        controller_factory: Optional[Callable[..., GlobalGPUController]] = None,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._starting_job_ids: set[str] = set()
        self._starting_params: Dict[str, Dict[str, Any]] = {}
        self._sessions_lock = threading.RLock()
        self._sessions_cond = threading.Condition(self._sessions_lock)
        self._controller_factory = controller_factory or GlobalGPUController
        atexit.register(self.shutdown)

    @staticmethod
    def _release_with_timeout(
        controller: GlobalGPUController,
        timeout_s: float = 10.0,
        on_late_result: Optional[Callable[[Optional[Exception]], None]] = None,
    ) -> bool:
        done = threading.Event()
        timed_out = threading.Event()
        callback_called = threading.Event()
        error_holder: Dict[str, Exception] = {}

        def _notify_late_result(error: Optional[Exception]) -> None:
            if on_late_result is None or callback_called.is_set():
                return
            callback_called.set()
            try:
                on_late_result(error)
            except Exception as exc:  # pragma: no cover - defensive callback guard
                logger.warning("Late release callback failed: %s", exc)

        def _release() -> None:
            error: Optional[Exception] = None
            try:
                controller.release()
            except Exception as exc:  # pragma: no cover - defensive
                error = exc
                error_holder["error"] = exc
            finally:
                done.set()
                if timed_out.is_set():
                    _notify_late_result(error)

        thread = threading.Thread(target=_release, daemon=True)
        thread.start()
        if not done.wait(timeout_s):
            timed_out.set()
            if done.is_set():
                _notify_late_result(error_holder.get("error"))
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
            gpu_ids: Visible GPU ordinals to target; None uses all visible GPUs.
            vram: Human-readable VRAM size to keep (for example, "1GiB").
            interval: Seconds between controller checks/actions.
            busy_threshold: Backoff threshold. Non-negative values back off when
                utilization is above this percent or telemetry is unavailable;
                ``-1`` disables utilization backoff for unconditional keepalive.
            job_id: Optional session identifier; a UUID is generated if omitted.

        Returns:
            Dict with the started session's job_id, e.g. ``{"job_id": "<id>"}``.

        Raises:
            ValueError: If the provided job_id already exists.
        """
        gpu_ids = validate_gpu_ids(gpu_ids)
        interval = validate_interval(interval)
        busy_threshold = validate_busy_threshold(busy_threshold)
        parse_vram_to_elements(vram)

        job_id = validate_job_id(job_id)
        if job_id is None:
            job_id = str(uuid.uuid4())
        params = {
            "gpu_ids": gpu_ids,
            "vram": vram,
            "interval": interval,
            "busy_threshold": busy_threshold,
        }
        with self._sessions_lock:
            if job_id in self._sessions or job_id in self._starting_job_ids:
                raise ValueError(f"job_id {job_id} already exists")
            self._starting_job_ids.add(job_id)
            self._starting_params[job_id] = params

        try:
            controller = self._controller_factory(
                gpu_ids=gpu_ids,
                interval=interval,
                vram_to_keep=vram,
                busy_threshold=busy_threshold,
            )
            controller.keep()
        except Exception:
            with self._sessions_lock:
                self._starting_job_ids.discard(job_id)
                self._starting_params.pop(job_id, None)
                self._sessions_cond.notify_all()
            raise

        with self._sessions_lock:
            self._starting_job_ids.discard(job_id)
            params = self._starting_params.pop(job_id, params)
            self._sessions[job_id] = Session(
                controller=controller,
                params=params,
            )
            self._sessions_cond.notify_all()
        logger.info("Started keep session %s on GPUs %s", job_id, gpu_ids)
        return {"job_id": job_id}

    @staticmethod
    def _empty_stop_result() -> Dict[str, Any]:
        return {"stopped": [], "timed_out": [], "failed": [], "errors": {}}

    @staticmethod
    def _timeout_error_message() -> str:
        return "Timed out while stopping session; release continues in background."

    @staticmethod
    def _finalize_stop_result(result: Dict[str, Any]) -> Dict[str, Any]:
        messages = []
        if result["timed_out"]:
            messages.append(
                "Timed out while stopping some sessions; release continues in background."
            )
        if result["failed"]:
            messages.append("Some sessions failed to stop; inspect status for details.")
        if messages:
            result["message"] = " ".join(messages)
        return result

    def _mark_session(
        self, job_id: str, session: Session, state: str, last_error: Optional[str]
    ) -> None:
        with self._sessions_lock:
            current = self._sessions.get(job_id)
            if current is session:
                current.state = state
                current.last_error = last_error

    def _mark_stop_timeout(self, job_id: str, session: Session) -> None:
        with self._sessions_lock:
            current = self._sessions.get(job_id)
            if current is session and current.state != "stop_failed":
                current.state = "stopping"
                current.last_error = self._timeout_error_message()

    def _finalize_late_release(
        self, job_id: str, session: Session, error: Optional[Exception]
    ) -> None:
        if error is None:
            with self._sessions_lock:
                if self._sessions.get(job_id) is session:
                    self._sessions.pop(job_id, None)
            logger.info("Completed delayed release for keep session %s", job_id)
            return

        message = str(error)
        self._mark_session(job_id, session, "stop_failed", message)
        logger.warning("Delayed release failed for keep session %s: %s", job_id, error)

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
            Dict with "stopped", "timed_out", "failed", and "errors" fields.
            If a specific job_id was not found, a "message" field explains the miss.
        """
        job_id = validate_job_id(job_id)
        if job_id is not None:
            with self._sessions_lock:
                while job_id in self._starting_job_ids:
                    self._sessions_cond.wait()
                session = self._sessions.get(job_id)
                if session and session.state == "stopping":
                    result = self._empty_stop_result()
                    result["timed_out"].append(job_id)
                    return self._finalize_stop_result(result)
                if session:
                    session.state = "stopping"
                    session.last_error = None
            if session:
                result = self._empty_stop_result()
                try:
                    released = self._release_with_timeout(
                        session.controller,
                        on_late_result=lambda error: self._finalize_late_release(
                            job_id, session, error
                        ),
                    )
                except Exception as exc:
                    error = str(exc)
                    self._mark_session(job_id, session, "stop_failed", error)
                    result["failed"].append(job_id)
                    result["errors"][job_id] = error
                    if not quiet:
                        logger.warning(
                            "Failed to stop keep session %s: %s", job_id, exc
                        )
                    return self._finalize_stop_result(result)
                if not quiet:
                    if released:
                        logger.info("Stopped keep session %s", job_id)
                    else:
                        logger.warning(
                            "Timed out while stopping keep session %s", job_id
                        )
                if released:
                    with self._sessions_lock:
                        if self._sessions.get(job_id) is session:
                            self._sessions.pop(job_id, None)
                    result["stopped"].append(job_id)
                    return self._finalize_stop_result(result)
                self._mark_stop_timeout(job_id, session)
                result["timed_out"].append(job_id)
                return self._finalize_stop_result(result)
            result = self._empty_stop_result()
            result["message"] = "job_id not found"
            return result

        with self._sessions_lock:
            starting_to_wait_for = set(self._starting_job_ids)
            while starting_to_wait_for & self._starting_job_ids:
                self._sessions_cond.wait()
            session_items = list(self._sessions.items())
            releasable_items = []
            release_outcomes: List[Dict[str, Any]] = [
                {} for _job_id, _session in session_items
            ]
            result = self._empty_stop_result()
            for index, (job_id, session) in enumerate(session_items):
                if session.state == "stopping":
                    release_outcomes[index] = {"state": "timed_out"}
                    continue
                session.state = "stopping"
                session.last_error = None
                releasable_items.append((index, job_id, session))

        def _release_one(index: int, job_id: str, session: Session) -> None:
            try:
                released = self._release_with_timeout(
                    session.controller,
                    on_late_result=lambda error, jid=job_id, sess=session: self._finalize_late_release(
                        jid, sess, error
                    ),
                )
            except Exception as exc:
                error = str(exc)
                self._mark_session(job_id, session, "stop_failed", error)
                release_outcomes[index] = {"state": "failed", "error": error}
                return
            if released:
                with self._sessions_lock:
                    if self._sessions.get(job_id) is session:
                        self._sessions.pop(job_id, None)
                release_outcomes[index] = {"state": "stopped"}
                return
            self._mark_stop_timeout(job_id, session)
            release_outcomes[index] = {"state": "timed_out"}

        release_threads = []
        for index, job_id, session in releasable_items:
            thread = threading.Thread(
                target=_release_one,
                args=(index, job_id, session),
            )
            thread.start()
            release_threads.append(thread)
        for thread in release_threads:
            thread.join()
        for (job_id, _session), outcome in zip(session_items, release_outcomes):
            state = outcome.get("state")
            if state == "stopped":
                result["stopped"].append(job_id)
            elif state == "timed_out":
                result["timed_out"].append(job_id)
            elif state == "failed":
                result["failed"].append(job_id)
                result["errors"][job_id] = outcome["error"]
        if result["stopped"] and not quiet:
            logger.info("Stopped sessions: %s", result["stopped"])
        if result["timed_out"] and not quiet:
            logger.warning("Timed out stopping sessions: %s", result["timed_out"])
        if result["failed"] and not quiet:
            logger.warning("Failed stopping sessions: %s", result["failed"])
        return self._finalize_stop_result(result)

    def status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        job_id = validate_job_id(job_id)
        if job_id is not None:
            with self._sessions_lock:
                session = self._sessions.get(job_id)
                if not session:
                    params = self._starting_params.get(job_id)
                    if params is not None:
                        return {
                            "active": True,
                            "job_id": job_id,
                            "params": params,
                            "state": "starting",
                            "last_error": None,
                        }
                    return {"active": False, "job_id": job_id}
                return {
                    "active": True,
                    "job_id": job_id,
                    "params": session.params,
                    "state": session.state,
                    "last_error": session.last_error,
                }
        with self._sessions_lock:
            return {
                "active_jobs": [
                    {
                        "job_id": jid,
                        "params": params,
                        "state": "starting",
                        "last_error": None,
                    }
                    for jid, params in self._starting_params.items()
                ]
                + [
                    {
                        "job_id": jid,
                        "params": sess.params,
                        "state": sess.state,
                        "last_error": sess.last_error,
                    }
                    for jid, sess in self._sessions.items()
                ],
            }

    def list_gpus(self) -> Dict[str, Any]:
        """Return detailed GPU info with visible and physical identifiers."""
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

    def _job_id_from_session_path(self, path: str) -> str:
        prefix = "/api/sessions/"
        if not path.startswith(prefix):
            raise ValueError("Missing job_id")
        encoded_job_id = path[len(prefix) :]
        if encoded_job_id == "" or encoded_job_id.endswith("/"):
            raise ValueError("Missing job_id")
        if "/" in encoded_job_id:
            raise ValueError("Invalid job_id path")
        job_id = unquote(encoded_job_id)
        validated = validate_job_id(job_id)
        if validated is None:
            raise ValueError("Missing job_id")
        return validated

    def _reject_session_route_components(self, parsed) -> bool:
        if parsed.params or parsed.query or parsed.fragment:
            self._json_response(
                400,
                {"error": {"message": "Invalid session path for job_id"}},
            )
            return True
        return False

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
            if self._reject_session_route_components(parsed):
                return
            self._json_response(200, server_ref.status())
            return
        if path.startswith("/api/sessions/"):
            if self._reject_session_route_components(parsed):
                return
            try:
                job_id = self._job_id_from_session_path(path)
            except ValueError as exc:
                self._json_response(400, {"error": {"message": str(exc)}})
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
                if self._reject_session_route_components(parsed):
                    return
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
                    validate_gpu_ids(gpu_ids)
                    visible_gpus = server_ref.list_gpus().get("gpus", [])
                    listed_ids = {
                        gpu["id"]
                        for gpu in visible_gpus
                        if isinstance(gpu.get("id"), int)
                    }
                    invalid_ids = [
                        gpu_id for gpu_id in gpu_ids if gpu_id not in listed_ids
                    ]
                    if visible_gpus and invalid_ids:
                        allowed_ids = ", ".join(
                            str(gpu_id) for gpu_id in sorted(listed_ids)
                        )
                        raise ValueError(
                            "gpu_ids must match listed visible GPU IDs"
                            f" ({allowed_ids}); got {invalid_ids}"
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
            if self._reject_session_route_components(parsed):
                return
            self._json_response(200, server_ref.stop_keep(job_id=None))
            return
        if path.startswith("/api/sessions/"):
            if self._reject_session_route_components(parsed):
                return
            try:
                job_id = self._job_id_from_session_path(path)
            except ValueError as exc:
                self._json_response(400, {"error": {"message": str(exc)}})
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
