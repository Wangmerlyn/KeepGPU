"""KeepGPU local service.

Supports MCP/JSON-RPC over stdio/HTTP and REST-style HTTP endpoints.

MCP protocol methods:
  - initialize
  - notifications/initialized
  - tools/list
  - tools/call

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
import ipaddress
import json
import sys
import uuid
import argparse
import threading
import time
import mimetypes
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer, ThreadingMixIn
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

from keep_gpu import __version__
from keep_gpu.global_gpu_controller.global_gpu_controller import (
    ControllerStartupUnavailable,
    GlobalGPUController,
    InvalidVisibleGPUSelectionError,
)
from keep_gpu.utilities.humanized_input import (
    PUBLIC_VRAM_MAX_BYTES,
    parse_vram_to_elements,
)
from keep_gpu.utilities.gpu_info import get_gpu_info
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    PUBLIC_INTERVAL_MAX_SECONDS,
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
    validate_job_id,
)

logger = setup_logger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_JSON_BODY_BYTES = 1_000_000
MCP_PROTOCOL_VERSION = "2025-06-18"
JSONRPC_PARSE_ERROR = -32700
JSONRPC_INVALID_REQUEST = -32600
JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602
JSONRPC_INTERNAL_ERROR = -32603
JSONRPC_STARTUP_UNAVAILABLE = -32000
MCP_ENDPOINT_HOST_ERROR = "host must be a DNS hostname or IPv4 address"
MCP_ENDPOINT_PORT_ERROR = "port must be an integer between 1 and 65535"
STARTUP_STOP_WAIT_TIMEOUT_SECONDS = 10.0

MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "start_keep",
        "title": "Start KeepGPU Session",
        "description": "Reserve VRAM on selected visible GPU ordinals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_ids": {
                    "type": ["array", "null"],
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": "Visible GPU ordinals; null or omitted uses all.",
                },
                "vram": {
                    "type": ["string", "integer"],
                    "minimum": 4,
                    "maximum": PUBLIC_VRAM_MAX_BYTES,
                    "default": "1GiB",
                    "description": "Human-readable VRAM amount or integer bytes to keep; byte-equivalent values must be no more than 1 PiB.",
                },
                "interval": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": PUBLIC_INTERVAL_MAX_SECONDS,
                    "default": 300,
                    "description": "Seconds between keep-alive checks.",
                },
                "busy_threshold": {
                    "type": "integer",
                    "minimum": -1,
                    "maximum": 100,
                    "default": DEFAULT_BUSY_THRESHOLD,
                    "description": "Defaults to 25; -1 disables utilization backoff; 0..100 backs off.",
                },
                "job_id": {
                    "type": ["string", "null"],
                    "description": "Optional URL-path-safe session identifier.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "stop_keep",
        "title": "Stop KeepGPU Session",
        "description": "Release one session by job_id, or all sessions when omitted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": ["string", "null"],
                    "description": "Session identifier; null or omitted stops all.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "status",
        "title": "Get KeepGPU Status",
        "description": "Return one session status or the active session list.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": ["string", "null"],
                    "description": "Session identifier; null or omitted lists all.",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_gpus",
        "title": "List Visible GPUs",
        "description": "List start-compatible visible GPU ordinals and metadata.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


@dataclass
class Session:
    controller: GlobalGPUController
    params: Dict[str, Any]
    state: str = "active"
    last_error: Optional[str] = None


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SessionInputError(ValueError):
    """Public session input validation error."""


class SessionStartupUnavailable(Exception):
    """Expected session startup failure caused by unavailable hardware/platform."""


def _validate_public_session_input(validator: Callable[..., Any], *args: Any) -> Any:
    try:
        return validator(*args)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SessionInputError(str(exc)) from exc


class KeepGPUServer:
    def __init__(
        self,
        controller_factory: Optional[Callable[..., GlobalGPUController]] = None,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._starting_job_ids: set[str] = set()
        self._starting_params: Dict[str, Dict[str, Any]] = {}
        self._pending_stop_job_ids: set[str] = set()
        self._startup_stop_wait_timeout_s = STARTUP_STOP_WAIT_TIMEOUT_SECONDS
        self._sessions_lock = threading.RLock()
        self._sessions_cond = threading.Condition(self._sessions_lock)
        self._controller_factory = controller_factory or GlobalGPUController
        atexit.register(self.shutdown)

    def _wait_for_starting_jobs_or_mark_pending(
        self, starting_snapshot: Dict[str, Dict[str, Any]]
    ) -> set[str]:
        deadline = time.monotonic() + self._startup_stop_wait_timeout_s
        with self._sessions_lock:
            while True:
                still_starting = {
                    job_id
                    for job_id, params in starting_snapshot.items()
                    if self._starting_params.get(job_id) is params
                }
                if not still_starting:
                    return set()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._pending_stop_job_ids.update(still_starting)
                    return still_starting
                self._sessions_cond.wait(remaining)

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
        interval: Union[int, float] = 300,
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a KeepGPU session that reserves VRAM on one or more GPUs.

        Args:
            gpu_ids: Visible GPU ordinals to target; None uses all visible GPUs.
            vram: Human-readable VRAM size to keep (for example, "1GiB").
            interval: Seconds between controller checks/actions.
            busy_threshold: Backoff threshold. Defaults to 25. Non-negative
                values back off when utilization is above this percent or
                telemetry is unavailable; ``-1`` disables utilization backoff
                for unconditional keepalive.
            job_id: Optional session identifier; a UUID is generated if omitted.

        Returns:
            Dict with the started session's job_id, e.g. ``{"job_id": "<id>"}``.

        Raises:
            ValueError: If the provided job_id already exists.
        """
        gpu_ids = _validate_public_session_input(validate_gpu_ids, gpu_ids)
        interval = _validate_public_session_input(validate_interval, interval)
        busy_threshold = _validate_public_session_input(
            validate_busy_threshold, busy_threshold
        )
        _validate_public_session_input(parse_vram_to_elements, vram)

        job_id = _validate_public_session_input(validate_job_id, job_id)
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
                raise SessionInputError(f"job_id {job_id} already exists")
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
        except Exception as exc:
            with self._sessions_lock:
                self._starting_job_ids.discard(job_id)
                self._starting_params.pop(job_id, None)
                self._pending_stop_job_ids.discard(job_id)
                self._sessions_cond.notify_all()
            if isinstance(exc, InvalidVisibleGPUSelectionError):
                raise SessionInputError(str(exc)) from exc
            if isinstance(exc, ControllerStartupUnavailable):
                raise SessionStartupUnavailable(str(exc)) from exc
            raise

        with self._sessions_lock:
            self._starting_job_ids.discard(job_id)
            params = self._starting_params.pop(job_id, params)
            pending_stop = job_id in self._pending_stop_job_ids
            self._pending_stop_job_ids.discard(job_id)
            session = Session(
                controller=controller,
                params=params,
                state="stopping" if pending_stop else "active",
                last_error=self._timeout_error_message() if pending_stop else None,
            )
            self._sessions[job_id] = session
            self._sessions_cond.notify_all()
        if pending_stop:
            threading.Thread(
                target=self._stop_current_session,
                kwargs={
                    "job_id": job_id,
                    "expected_session": session,
                    "pending_stop_cleanup": True,
                },
                daemon=True,
            ).start()
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

    @staticmethod
    def _refresh_session_runtime_state(session: Session) -> None:
        if session.state != "active":
            return
        runtime_error = getattr(session.controller, "runtime_error", None)
        if not callable(runtime_error):
            return
        error = runtime_error()
        if error is not None:
            session.state = "runtime_failed"
            session.last_error = str(error)

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

    def _stop_current_session(
        self,
        job_id: str,
        quiet: bool = False,
        expected_session: Optional[Session] = None,
        pending_stop_cleanup: bool = False,
    ) -> Optional[Dict[str, Any]]:
        with self._sessions_lock:
            session = self._sessions.get(job_id)
            if expected_session is not None and session is not expected_session:
                return None
            if session is None:
                return None
            release_existing_stopping = (
                pending_stop_cleanup and expected_session is session
            )
            if session.state == "stopping" and not release_existing_stopping:
                result = self._empty_stop_result()
                result["timed_out"].append(job_id)
                return self._finalize_stop_result(result)
            if session.state != "stopping":
                session.state = "stopping"
                session.last_error = None

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
                logger.warning("Failed to stop keep session %s: %s", job_id, exc)
            return self._finalize_stop_result(result)
        if not quiet:
            if released:
                logger.info("Stopped keep session %s", job_id)
            else:
                logger.warning("Timed out while stopping keep session %s", job_id)
        if released:
            with self._sessions_lock:
                if self._sessions.get(job_id) is session:
                    self._sessions.pop(job_id, None)
            result["stopped"].append(job_id)
            return self._finalize_stop_result(result)
        self._mark_stop_timeout(job_id, session)
        result["timed_out"].append(job_id)
        return self._finalize_stop_result(result)

    def _starting_status(self, job_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        pending_stop = job_id in self._pending_stop_job_ids
        return {
            "job_id": job_id,
            "params": params,
            "state": "stopping" if pending_stop else "starting",
            "last_error": self._timeout_error_message() if pending_stop else None,
        }

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
        job_id = _validate_public_session_input(validate_job_id, job_id)
        if job_id is not None:
            with self._sessions_lock:
                starting_params = self._starting_params.get(job_id)
                expected_session = self._sessions.get(job_id)
                starting_snapshot = (
                    {job_id: starting_params} if starting_params is not None else {}
                )
            timed_out_starting = self._wait_for_starting_jobs_or_mark_pending(
                starting_snapshot
            )
            if job_id in timed_out_starting:
                result = self._empty_stop_result()
                result["timed_out"].append(job_id)
                return self._finalize_stop_result(result)
            if starting_params is not None:
                with self._sessions_lock:
                    expected_session = self._sessions.get(job_id)
                    if (
                        expected_session is not None
                        and expected_session.params is not starting_params
                    ):
                        expected_session = None
            if expected_session is None:
                result = self._empty_stop_result()
                result["message"] = "job_id not found"
                return result
            stop_result = self._stop_current_session(
                job_id, quiet=quiet, expected_session=expected_session
            )
            if stop_result is not None:
                return stop_result
            result = self._empty_stop_result()
            result["message"] = "job_id not found"
            return result

        with self._sessions_lock:
            starting_snapshot = dict(self._starting_params)
            initial_session_items = list(self._sessions.items())
        timed_out_starting = self._wait_for_starting_jobs_or_mark_pending(
            starting_snapshot
        )
        with self._sessions_lock:
            session_items = [
                (job_id, session)
                for job_id, session in initial_session_items
                if self._sessions.get(job_id) is session
            ]
            session_items.extend(
                (job_id, session)
                for job_id, params in starting_snapshot.items()
                if job_id not in timed_out_starting
                for session in [self._sessions.get(job_id)]
                if session is not None and session.params is params
            )
            releasable_items = []
            release_outcomes: List[Dict[str, Any]] = [
                {} for _job_id, _session in session_items
            ]
            result = self._empty_stop_result()
            for starting_job_id in starting_snapshot:
                if starting_job_id in timed_out_starting:
                    result["timed_out"].append(starting_job_id)
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
        job_id = _validate_public_session_input(validate_job_id, job_id)
        if job_id is not None:
            with self._sessions_lock:
                session = self._sessions.get(job_id)
                if not session:
                    params = self._starting_params.get(job_id)
                    if params is not None:
                        status = self._starting_status(job_id, params)
                        status["active"] = True
                        return status
                    return {"active": False, "job_id": job_id}
                self._refresh_session_runtime_state(session)
                return {
                    "active": True,
                    "job_id": job_id,
                    "params": session.params,
                    "state": session.state,
                    "last_error": session.last_error,
                }
        with self._sessions_lock:
            for session in self._sessions.values():
                self._refresh_session_runtime_state(session)
            return {
                "active_jobs": [
                    self._starting_status(jid, params)
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


def _jsonrpc_result(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _jsonrpc_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _is_valid_jsonrpc_id(req_id: Any) -> bool:
    return isinstance(req_id, (str, int)) and not isinstance(req_id, bool)


_DIRECT_METHOD_PARAMS = {
    "start_keep": {"gpu_ids", "vram", "interval", "busy_threshold", "job_id"},
    "stop_keep": {"job_id"},
    "status": {"job_id"},
    "list_gpus": set(),
}
_REST_SESSION_FIELDS = _DIRECT_METHOD_PARAMS["start_keep"]


def _validate_direct_method_params(method: str, params: Dict[str, Any]) -> None:
    allowed = _DIRECT_METHOD_PARAMS.get(method)
    if allowed is None:
        return
    unknown = sorted(set(params) - allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise JSONRPCError(
            JSONRPC_INVALID_PARAMS,
            f"Unknown params for {method}: {joined}",
        )


def _call_keepgpu_method(
    server: KeepGPUServer, method: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    _validate_direct_method_params(method, params)
    try:
        if method == "start_keep":
            return server.start_keep(**params)
        if method == "stop_keep":
            return server.stop_keep(**params)
        if method == "status":
            return server.status(**params)
        if method == "list_gpus":
            return server.list_gpus()
    except SessionInputError as exc:
        raise JSONRPCError(JSONRPC_INVALID_PARAMS, str(exc)) from exc
    except SessionStartupUnavailable as exc:
        raise JSONRPCError(JSONRPC_STARTUP_UNAVAILABLE, str(exc)) from exc
    raise JSONRPCError(JSONRPC_METHOD_NOT_FOUND, f"Unknown method: {method}")


def _mcp_initialize_result(params: Dict[str, Any]) -> Dict[str, Any]:
    protocol_version = params.get("protocolVersion") or MCP_PROTOCOL_VERSION
    if protocol_version != MCP_PROTOCOL_VERSION:
        protocol_version = MCP_PROTOCOL_VERSION
    return {
        "protocolVersion": protocol_version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {
            "name": "keepgpu",
            "title": "KeepGPU",
            "version": __version__,
        },
        "instructions": (
            "Use start_keep to reserve VRAM only when needed, status to inspect "
            "sessions, stop_keep to release sessions, and list_gpus to choose "
            "visible GPU ordinals."
        ),
    }


def _mcp_tool_result(payload: Any, is_error: bool = False) -> Dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True)
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _mcp_call_tool(server: KeepGPUServer, params: Dict[str, Any]) -> Dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise JSONRPCError(
            JSONRPC_INVALID_PARAMS, "Tool call requires a non-empty tool name."
        )
    if not isinstance(arguments, dict):
        raise JSONRPCError(
            JSONRPC_INVALID_PARAMS, "Tool call arguments must be an object."
        )
    if name not in {tool["name"] for tool in MCP_TOOLS}:
        raise JSONRPCError(JSONRPC_INVALID_PARAMS, f"Unknown tool: {name}")
    try:
        return _mcp_tool_result(_call_keepgpu_method(server, name, arguments))
    except JSONRPCError as exc:
        if exc.code not in (JSONRPC_INVALID_PARAMS, JSONRPC_STARTUP_UNAVAILABLE):
            raise
        return _mcp_tool_result(str(exc), True)


def _prevalidate_rest_session_payload(
    server: KeepGPUServer, payload: Any
) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    unknown_fields = set(payload) - _REST_SESSION_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown request fields: {sorted(unknown_fields)}")

    safe_payload = dict(payload)
    if "gpu_ids" in safe_payload:
        safe_payload["gpu_ids"] = _validate_public_session_input(
            validate_gpu_ids, safe_payload["gpu_ids"]
        )
    if "interval" in safe_payload:
        safe_payload["interval"] = _validate_public_session_input(
            validate_interval, safe_payload["interval"]
        )
    if "busy_threshold" in safe_payload:
        safe_payload["busy_threshold"] = _validate_public_session_input(
            validate_busy_threshold, safe_payload["busy_threshold"]
        )
    if "vram" in safe_payload:
        _validate_public_session_input(parse_vram_to_elements, safe_payload["vram"])
    if "job_id" in safe_payload:
        job_id = _validate_public_session_input(validate_job_id, safe_payload["job_id"])
        safe_payload["job_id"] = job_id
        if job_id is not None:
            with server._sessions_lock:
                if job_id in server._sessions or job_id in server._starting_job_ids:
                    raise SessionInputError(f"job_id {job_id} already exists")
    return safe_payload


def _handle_request(server: KeepGPUServer, payload: Any) -> Optional[Dict[str, Any]]:
    """
    Dispatch a JSON-RPC payload to the server and return a response dict.

    Args:
        server: Target KeepGPUServer.
        payload: Dict with "method", optional "params", and optional "id".

    Returns:
        JSON-RPC-style dict containing either "result" or "error" plus "id",
        or None for JSON-RPC notifications that do not expect a response.
    """
    req_id = None
    try:
        if not isinstance(payload, dict):
            raise JSONRPCError(
                JSONRPC_INVALID_REQUEST, "JSON-RPC messages must be objects."
            )
        raw_id = payload.get("id")
        if "id" in payload and _is_valid_jsonrpc_id(raw_id):
            req_id = raw_id
        method = payload.get("method")
        params = payload.get("params", {})
        if not isinstance(method, str) or not method:
            raise JSONRPCError(JSONRPC_INVALID_REQUEST, "Request method is required.")
        if method.startswith("notifications/") and "id" not in payload:
            return None
        if "jsonrpc" in payload and payload["jsonrpc"] != "2.0":
            raise JSONRPCError(JSONRPC_INVALID_REQUEST, "JSON-RPC version must be 2.0.")
        if "id" not in payload or not _is_valid_jsonrpc_id(raw_id):
            raise JSONRPCError(JSONRPC_INVALID_REQUEST, "Requests must include an id.")
        if method.startswith("notifications/"):
            raise JSONRPCError(
                JSONRPC_INVALID_REQUEST, "Notifications must not include an id."
            )
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JSONRPCError(JSONRPC_INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            result = _mcp_initialize_result(params)
        elif method == "tools/list":
            result = {"tools": MCP_TOOLS}
        elif method == "tools/call":
            result = _mcp_call_tool(server, params)
        else:
            result = _call_keepgpu_method(server, method, params)
        return _jsonrpc_result(req_id, result)
    except JSONRPCError as exc:
        return _jsonrpc_error(req_id, exc.code, exc.message)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Request failed")
        return _jsonrpc_error(req_id, JSONRPC_INTERNAL_ERROR, str(exc))


class _JSONRPCHandler(BaseHTTPRequestHandler):
    server_version = "KeepGPU-MCP/0.1"

    def _empty_response(self, status: int) -> None:
        self.send_response(status)
        self.send_header("content-length", "0")
        self.end_headers()

    def _json_response(
        self,
        status: int,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        write_body: bool = True,
    ) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        if headers is not None:
            for name, value in headers.items():
                self.send_header(name, value)
        self.end_headers()
        if write_body:
            self.wfile.write(data)

    @staticmethod
    def _allowed_methods_for_path(path: str) -> Optional[Tuple[str, ...]]:
        if path == "/health":
            return ("GET",)
        if path == "/api/gpus":
            return ("GET",)
        if path == "/api/sessions":
            return ("GET", "POST", "DELETE")
        if path.startswith("/api/sessions/"):
            encoded_job_id = path[len("/api/sessions/") :]
            if encoded_job_id == "" or "/" in encoded_job_id:
                return None
            return ("GET", "DELETE")
        if path == "/":
            return ("GET", "POST")
        if path == "/rpc":
            return ("POST",)
        return None

    def _send_api_rpc_unsupported_method_response(self) -> bool:
        if not hasattr(self, "path") or not hasattr(self, "command"):
            return False
        parsed = urlparse(self.path)
        path = parsed.path
        write_body = self.command != "HEAD"
        allowed_methods = self._allowed_methods_for_path(path)
        if allowed_methods is not None:
            self._json_response(
                405,
                {"error": {"message": "Method not allowed"}},
                headers={"Allow": ", ".join(allowed_methods)},
                write_body=write_body,
            )
            return True
        if path.startswith("/api/"):
            self._json_response(
                404,
                {"error": {"message": "Unknown endpoint"}},
                write_body=write_body,
            )
            return True
        return False

    def _send_known_route_unsupported_method_response(self, path: str) -> bool:
        allowed_methods = self._allowed_methods_for_path(path)
        if allowed_methods is None or self.command in allowed_methods:
            return False
        return self._send_api_rpc_unsupported_method_response()

    def send_error(self, code, message=None, explain=None):  # noqa: ANN001, N802
        if int(code) == 501 and self._send_api_rpc_unsupported_method_response():
            return
        super().send_error(code, message, explain)

    def _json_runtime_error(self, method: str, path: str, exc: Exception) -> None:
        logger.exception("%s request failed for path %s", method, path)
        self._json_response(
            500,
            {
                "error": {
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                }
            },
        )

    def _read_json_body(self) -> Any:
        raw_length = self.headers.get("content-length", "0")
        try:
            length = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise ValueError("Content-Length must be a non-negative integer") from exc
        if length < 0:
            raise ValueError("Content-Length must be a non-negative integer")
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
        try:
            if path == "/rpc":
                self._send_api_rpc_unsupported_method_response()
                return

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
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            self._json_runtime_error("GET", path, exc)

    def do_POST(self):  # noqa: N802
        """
        Handle HTTP JSON-RPC and REST session POST requests.

        JSON-RPC parse failures return JSON-RPC parse-error envelopes. REST
        session parse failures return HTTP 400 with a structured error object.
        """
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if self._send_known_route_unsupported_method_response(path):
                return
            if path not in ("/api/sessions", "/", "/rpc"):
                self._json_response(404, {"error": {"message": "Unknown endpoint"}})
                return

            server_ref = self.server.keepgpu_server  # type: ignore[attr-defined]
            if path == "/api/sessions" and self._reject_session_route_components(
                parsed
            ):
                return

            try:
                payload = self._read_json_body()
            except (
                json.JSONDecodeError,
                ValueError,
                UnicodeDecodeError,
                TypeError,
            ) as exc:
                if path in ("/", "/rpc"):
                    self._json_response(
                        200, _jsonrpc_error(None, JSONRPC_PARSE_ERROR, str(exc))
                    )
                    return
                self._json_response(400, {"error": {"message": f"Bad request: {exc}"}})
                return

            if path == "/api/sessions":
                try:
                    safe_payload = _prevalidate_rest_session_payload(
                        server_ref, payload
                    )
                    gpu_ids = safe_payload.get("gpu_ids")
                except (ValueError, TypeError) as exc:
                    self._json_response(
                        400, {"error": {"message": f"Bad request: {exc}"}}
                    )
                    return

                if gpu_ids is not None:
                    visible_gpus = server_ref.list_gpus().get("gpus", [])
                    listed_ids = {
                        gpu["id"]
                        for gpu in visible_gpus
                        if isinstance(gpu.get("id"), int)
                    }
                    invalid_ids = [
                        gpu_id for gpu_id in gpu_ids if gpu_id not in listed_ids
                    ]
                    if invalid_ids:
                        allowed_ids = (
                            ", ".join(str(gpu_id) for gpu_id in sorted(listed_ids))
                            or "none"
                        )
                        self._json_response(
                            400,
                            {
                                "error": {
                                    "message": (
                                        "Bad request: gpu_ids must match listed "
                                        f"visible GPU IDs ({allowed_ids}); got "
                                        f"{invalid_ids}"
                                    )
                                }
                            },
                        )
                        return

                try:
                    result = server_ref.start_keep(**safe_payload)
                except SessionInputError as exc:
                    self._json_response(
                        400, {"error": {"message": f"Bad request: {exc}"}}
                    )
                    return
                except SessionStartupUnavailable as exc:
                    self._json_response(
                        503,
                        {
                            "error": {
                                "message": str(exc),
                                "type": exc.__class__.__name__,
                            }
                        },
                    )
                    return
                self._json_response(200, result)
                return

            # JSON-RPC compatibility endpoint.
            if path in ("/", "/rpc"):
                response = _handle_request(server_ref, payload)
                if response is None:
                    self._empty_response(202)
                    return
                self._json_response(200, response)
                return
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            self._json_runtime_error("POST", path, exc)

    def do_DELETE(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if self._send_known_route_unsupported_method_response(path):
                return
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
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            self._json_runtime_error("DELETE", path, exc)

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
        except json.JSONDecodeError as exc:
            response = _jsonrpc_error(None, JSONRPC_PARSE_ERROR, str(exc))
        except Exception as exc:
            response = _jsonrpc_error(None, JSONRPC_INTERNAL_ERROR, str(exc))
        if response is None:
            continue
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


def _validate_mcp_http_host(host: str) -> str:
    def _is_dns_hostname(value: str) -> bool:
        if len(value) > 253 or value.endswith("."):
            return False
        labels = value.split(".")
        if labels[-1].isdigit():
            return False
        for label in labels:
            if not 1 <= len(label) <= 63:
                return False
            if label.startswith("-") or label.endswith("-"):
                return False
            if not all(
                char.isascii() and (char.isalnum() or char == "-") for char in label
            ):
                return False
        return True

    if not isinstance(host, str) or host.strip() != host or not host:
        raise ValueError(MCP_ENDPOINT_HOST_ERROR)
    try:
        parsed_ip = ipaddress.ip_address(host)
    except ValueError:
        if not _is_dns_hostname(host):
            raise ValueError(MCP_ENDPOINT_HOST_ERROR) from None
    else:
        if parsed_ip.version != 4:
            raise ValueError(MCP_ENDPOINT_HOST_ERROR)
    return host


def _validate_mcp_http_port(port: int) -> int:
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError(MCP_ENDPOINT_PORT_ERROR)
    return port


def _validate_mcp_http_endpoint(host: str, port: int) -> tuple[str, int]:
    return _validate_mcp_http_host(host), _validate_mcp_http_port(port)


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

    if args.mode == "http":
        try:
            args.host, args.port = _validate_mcp_http_endpoint(args.host, args.port)
        except ValueError as exc:
            parser.error(str(exc))

    server = KeepGPUServer()
    if args.mode == "stdio":
        run_stdio(server)
    else:
        run_http(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
