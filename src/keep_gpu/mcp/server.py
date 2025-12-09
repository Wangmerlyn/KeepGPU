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
        """
        Start a KeepGPU session that periodically reserves the specified amount of VRAM on one or more GPUs.
        
        Parameters:
            gpu_ids (Optional[List[int]]): List of GPU indices to target; if `None`, all available GPUs may be considered.
            vram (str): Amount of VRAM to reserve (human-readable, e.g. "1GiB").
            interval (int): Time in seconds between controller checks/actions.
            busy_threshold (int): Numeric threshold controlling what the controller treats as "busy" (semantics provided by the controller).
            job_id (Optional[str]): Identifier for the session; when `None`, a new UUID is generated.
        
        Returns:
            dict: A dictionary with the started session's `job_id`, e.g. `{"job_id": "<id>"}`.
        
        Raises:
            ValueError: If `job_id` is provided and already exists.
        """
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
        """
        Stop one or all active keep sessions.
        
        If `job_id` is provided, stops and removes that session if it exists; otherwise stops and removes all sessions. When `quiet` is True, informational logging about stopped sessions is suppressed.
        
        Parameters:
            job_id (Optional[str]): Identifier of the session to stop. If omitted, all sessions are stopped.
            quiet (bool): If True, do not emit informational logs about stopped sessions.
        
        Returns:
            result (Dict[str, Any]): A dictionary with a "stopped" key listing stopped job IDs. If a specific
            `job_id` was requested but not found, the dictionary also includes a "message" explaining that.
        """
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
        """
        Stop all active sessions and release resources, suppressing any errors that occur during interpreter teardown.
        
        This attempts to stop every session (quietly) and ignores exceptions to avoid noisy errors when the interpreter is shutting down.
        """
        try:
            self.stop_keep(None, quiet=True)
        except Exception:  # pragma: no cover - defensive
            # Avoid noisy errors during interpreter teardown
            return


def _handle_request(server: KeepGPUServer, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches a JSON-RPC-like request payload to the corresponding KeepGPUServer method and returns a JSON-RPC response object.
    
    Parameters:
        server (KeepGPUServer): The server instance whose methods will be invoked.
        payload (dict): The incoming request object; expected keys:
            - "method" (str): RPC method name ("start_keep", "stop_keep", "status", "list_gpus").
            - "params" (dict, optional): Keyword arguments for the method.
            - "id" (any, optional): Caller-provided request identifier preserved in the response.
    
    Returns:
        dict: A JSON-RPC-style response containing:
            - "id": the original request id (or None if not provided).
            - "result": the method's return value on success.
            - OR "error": an object with a "message" string describing the failure.
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

    def do_POST(self):  # noqa: N802
        """
        Handle HTTP POST requests containing a JSON-RPC payload and send a JSON response.
        
        Reads the request body using the Content-Length header, parses it as JSON, dispatches the payload to the internal JSON-RPC dispatcher, and writes the dispatcher result as an application/json response. If the request body cannot be decoded or parsed, responds with HTTP 400 and a JSON error object describing the parsing error.
        """
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
        """
        Suppress the BaseHTTPRequestHandler's default request logging by overriding log_message to do nothing.
        
        Parameters:
            format (str): The format string provided by BaseHTTPRequestHandler.
            *args: Values to interpolate into `format`.
        """
        return


def run_stdio(server: KeepGPUServer) -> None:
    """
    Read line-delimited JSON-RPC requests from stdin, dispatch each request to the server, and write the JSON response to stdout.
    
    Parameters:
        server (KeepGPUServer): Server instance used to handle JSON-RPC requests.
    
    Description:
        - Processes each non-empty line from stdin as a JSON payload.
        - On successful handling, writes the JSON-RPC response followed by a newline to stdout and flushes.
        - If parsing or handling raises an exception, writes an error object with the exception message as the response.
    """
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
    """
    Start a lightweight HTTP JSON-RPC server that exposes the given KeepGPUServer on the specified host and port.
    
    Starts a TCP HTTP server serving _JSONRPCHandler in a background thread, logs the listening address, waits for the thread to finish, and on interruption or shutdown performs a clean shutdown of the HTTP server and calls server.shutdown() to release resources.
    
    Parameters:
        server (KeepGPUServer): The KeepGPUServer instance whose RPC methods will be exposed over HTTP.
        host (str): Host address to bind the HTTP server to.
        port (int): TCP port to bind the HTTP server to.
    """
    class _Server(TCPServer):
        allow_reuse_address = True

    httpd = _Server((host, port), _JSONRPCHandler)
    httpd.keepgpu_server = server  # type: ignore[attr-defined]

    def _serve():
        """
        Run the HTTP server's request loop until the server is shut down.
        
        Blocks the current thread and processes incoming HTTP requests for the
        server instance until the server is stopped.
        """
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
    """
    Entry point for the KeepGPU MCP server that parses command-line arguments and starts the chosen transport.
    
    Parses --mode (stdio or http), --host and --port (for http mode), instantiates a KeepGPUServer, and runs either the stdio loop or the HTTP server based on the selected mode.
    """
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