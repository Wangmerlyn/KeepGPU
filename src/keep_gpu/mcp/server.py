"""
Minimal MCP-style JSON-RPC server for KeepGPU.

Run over stdin/stdout (default) or a lightweight HTTP server.
Supported methods:
  - start_keep(gpu_ids, vram, interval, busy_threshold, job_id)
  - stop_keep(job_id=None)  # None stops all
  - status(job_id=None)     # None lists all
  - list_gpus()
"""

from __future__ import annotations

import atexit
import json
import sys
import uuid
import argparse
import threading
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities.gpu_info import get_gpu_info
from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


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
        self._controller_factory = controller_factory or GlobalGPUController
        atexit.register(self.shutdown)

    def start_keep(
        self,
        gpu_ids: Optional[List[int]] = None,
        vram: str = "1GiB",
        interval: int = 300,
        busy_threshold: int = -1,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_id = job_id or str(uuid.uuid4())
        if job_id in self._sessions:
            raise ValueError(f"job_id {job_id} already exists")

        controller = self._controller_factory(
            gpu_ids=gpu_ids,
            interval=interval,
            vram_to_keep=vram,
            busy_threshold=busy_threshold,
        )
        controller.keep()
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
        if job_id:
            session = self._sessions.pop(job_id, None)
            if session:
                session.controller.release()
                if not quiet:
                    logger.info("Stopped keep session %s", job_id)
                return {"stopped": [job_id]}
            return {"stopped": [], "message": "job_id not found"}

        stopped_ids = list(self._sessions.keys())
        for job_id in stopped_ids:
            session = self._sessions.pop(job_id)
            session.controller.release()
        if stopped_ids and not quiet:
            logger.info("Stopped sessions: %s", stopped_ids)
        return {"stopped": stopped_ids}

    def status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        if job_id:
            session = self._sessions.get(job_id)
            if not session:
                return {"active": False, "job_id": job_id}
            return {
                "active": True,
                "job_id": job_id,
                "params": session.params,
            }
        return {
            "active_jobs": [
                {"job_id": jid, "params": sess.params}
                for jid, sess in self._sessions.items()
            ]
        }

    def list_gpus(self) -> Dict[str, Any]:
        """Return detailed GPU info (id, name, memory, utilization)."""
        infos = get_gpu_info()
        return {"gpus": infos}

    def shutdown(self) -> None:
        try:
            self.stop_keep(None, quiet=True)
        except Exception:  # pragma: no cover - defensive
            # Avoid noisy errors during interpreter teardown
            return


def _handle_request(server: KeepGPUServer, payload: Dict[str, Any]) -> Dict[str, Any]:
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

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            payload = json.loads(body)
            response = _handle_request(self.server.keepgpu_server, payload)  # type: ignore[attr-defined]
            status = 200
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
            response = {"error": {"message": f"Bad request: {exc}"}}
            status = 400
        data = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A003
        return


def run_stdio(server: KeepGPUServer) -> None:
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
    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server((host, port), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]

    def _serve():
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
