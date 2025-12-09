"""
Minimal MCP-style JSON-RPC server for KeepGPU.

The server reads JSON lines from stdin and writes JSON responses to stdout.
Supported methods:
  - start_keep(gpu_ids, vram, interval, busy_threshold, job_id)
  - stop_keep(job_id=None)  # None stops all
  - status(job_id=None)     # None lists all
"""

from __future__ import annotations

import atexit
import json
import sys
import uuid
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

    def stop_keep(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        if job_id:
            session = self._sessions.pop(job_id, None)
            if session:
                session.controller.release()
                logger.info("Stopped keep session %s", job_id)
                return {"stopped": [job_id]}
            return {"stopped": [], "message": "job_id not found"}

        stopped_ids = list(self._sessions.keys())
        for job_id in stopped_ids:
            session = self._sessions.pop(job_id)
            session.controller.release()
        if stopped_ids:
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
                {"job_id": jid, "params": sess.params} for jid, sess in self._sessions.items()
            ]
        }

    def list_gpus(self) -> Dict[str, Any]:
        """Return detailed GPU info (id, name, memory, utilization)."""
        infos = get_gpu_info()
        return {"gpus": infos}

    def shutdown(self) -> None:
        try:
            self.stop_keep(None)
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


def main() -> None:
    server = KeepGPUServer()
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


if __name__ == "__main__":
    main()
