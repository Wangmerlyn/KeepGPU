import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from keep_gpu.mcp.server import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    KeepGPUServer,
    _handle_request,
)
from keep_gpu.utilities.humanized_input import PUBLIC_VRAM_MAX_BYTES
from keep_gpu.utilities import platform_manager as pm


class DummyController:
    def __init__(self, gpu_ids=None, interval=0, vram_to_keep=None, busy_threshold=0):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.vram_to_keep = vram_to_keep
        self.busy_threshold = busy_threshold
        self.kept = False
        self.released = False

    def keep(self):
        self.kept = True

    def release(self):
        self.released = True


def dummy_factory(**kwargs):
    return DummyController(**kwargs)


def make_server() -> KeepGPUServer:
    return KeepGPUServer(controller_factory=cast(Any, dummy_factory))


def _wait_until(condition, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_start_status_stop_cycle():
    server = make_server()
    res = server.start_keep(gpu_ids=[1], vram="2GiB", interval=5, busy_threshold=20)
    job_id = res["job_id"]

    status = server.status(job_id)
    assert status["active"]
    assert status["params"]["gpu_ids"] == [1]
    assert status["params"]["vram"] == "2GiB"
    assert status["params"]["interval"] == 5
    assert status["params"]["busy_threshold"] == 20

    stopped = server.stop_keep(job_id)
    assert job_id in stopped["stopped"]
    assert server.status(job_id)["active"] is False


def test_start_keep_preserves_fractional_interval():
    server = make_server()

    res = server.start_keep(gpu_ids=[0], interval=0.5)
    job_id = res["job_id"]

    assert server.status(job_id)["params"]["interval"] == 0.5
    assert server._sessions[job_id].controller.interval == 0.5


def test_start_keep_defaults_to_eco_safe_busy_threshold():
    server = make_server()

    res = server.start_keep(job_id="default-threshold", gpu_ids=[0])
    job_id = res["job_id"]

    status = server.status(job_id)
    assert status["params"]["busy_threshold"] == 25
    controller = server._sessions[job_id].controller
    assert controller.busy_threshold == 25


def test_start_keep_preserves_explicit_unconditional_busy_threshold():
    server = make_server()

    res = server.start_keep(
        job_id="unconditional-threshold",
        gpu_ids=[0],
        busy_threshold=-1,
    )
    job_id = res["job_id"]

    assert server.status(job_id)["params"]["busy_threshold"] == -1


def test_status_marks_active_session_runtime_failed_when_controller_reports_error():
    class RuntimeFailedController(DummyController):
        def runtime_error(self):
            return RuntimeError("rank 0: allocation retries exhausted")

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: RuntimeFailedController(**kwargs))
    )
    job_id = server.start_keep(job_id="runtime-failure", gpu_ids=[0])["job_id"]

    status = server.status(job_id)

    assert status["active"] is True
    assert status["state"] == "runtime_failed"
    assert status["last_error"] == "rank 0: allocation retries exhausted"
    assert server._sessions[job_id].controller.released is False


def test_status_list_marks_runtime_failed_sessions():
    class RuntimeFailedController(DummyController):
        def runtime_error(self):
            return RuntimeError("rank 0: allocation retries exhausted")

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: RuntimeFailedController(**kwargs))
    )
    job_id = server.start_keep(job_id="runtime-failure", gpu_ids=[0])["job_id"]

    status = server.status()

    assert status["active_jobs"] == [
        {
            "job_id": job_id,
            "params": {
                "gpu_ids": [0],
                "vram": "1GiB",
                "interval": 300,
                "busy_threshold": 25,
            },
            "state": "runtime_failed",
            "last_error": "rank 0: allocation retries exhausted",
        }
    ]


def test_status_retains_first_runtime_failure_without_refreshing_again():
    class RuntimeFailedController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.runtime_error_calls = 0

        def runtime_error(self):
            self.runtime_error_calls += 1
            return RuntimeError(f"failure {self.runtime_error_calls}")

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: RuntimeFailedController(**kwargs))
    )
    job_id = server.start_keep(job_id="runtime-failure", gpu_ids=[0])["job_id"]
    controller = server._sessions[job_id].controller

    first_status = server.status(job_id)
    second_status = server.status(job_id)

    assert first_status["state"] == "runtime_failed"
    assert first_status["last_error"] == "failure 1"
    assert second_status["state"] == "runtime_failed"
    assert second_status["last_error"] == "failure 1"
    assert controller.runtime_error_calls == 1


def test_status_runtime_health_does_not_overwrite_retained_stop_states():
    class RuntimeFailedController(DummyController):
        def runtime_error(self):
            return RuntimeError("rank 0: allocation retries exhausted")

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: RuntimeFailedController(**kwargs))
    )
    stopping_job = server.start_keep(job_id="stopping-job", gpu_ids=[0])["job_id"]
    stop_failed_job = server.start_keep(job_id="stop-failed-job", gpu_ids=[1])["job_id"]

    server._sessions[stopping_job].state = "stopping"
    server._sessions[stopping_job].last_error = "release still running"
    server._sessions[stop_failed_job].state = "stop_failed"
    server._sessions[stop_failed_job].last_error = "release failed"

    assert server.status(stopping_job)["state"] == "stopping"
    assert server.status(stopping_job)["last_error"] == "release still running"
    assert server.status(stop_failed_job)["state"] == "stop_failed"
    assert server.status(stop_failed_job)["last_error"] == "release failed"

    jobs = {job["job_id"]: job for job in server.status()["active_jobs"]}
    assert jobs[stopping_job]["state"] == "stopping"
    assert jobs[stop_failed_job]["state"] == "stop_failed"


def test_status_reports_starting_session_during_controller_keep():
    keep_started = threading.Event()
    keep_release = threading.Event()
    result_holder = {}
    error_holder = {}

    class BlockingStartController(DummyController):
        def keep(self):
            self.kept = True
            keep_started.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )

    def start_session():
        try:
            result_holder["result"] = server.start_keep(
                job_id="starting-job",
                gpu_ids=[0],
                vram="512MB",
                interval=7,
                busy_threshold=25,
            )
        except Exception as exc:  # pragma: no cover - test failure helper
            error_holder["error"] = exc

    start_thread = threading.Thread(target=start_session)
    start_thread.start()
    try:
        assert keep_started.wait(timeout=1.0)

        expected_params = {
            "gpu_ids": [0],
            "vram": "512MB",
            "interval": 7,
            "busy_threshold": 25,
        }
        assert server.status("starting-job") == {
            "active": True,
            "job_id": "starting-job",
            "params": expected_params,
            "state": "starting",
            "last_error": None,
        }
        assert server.status()["active_jobs"] == [
            {
                "job_id": "starting-job",
                "params": expected_params,
                "state": "starting",
                "last_error": None,
            }
        ]
    finally:
        keep_release.set()

    start_thread.join(timeout=1.0)
    assert not start_thread.is_alive()
    assert error_holder == {}
    assert result_holder["result"] == {"job_id": "starting-job"}
    assert server.status("starting-job")["state"] == "active"


def test_start_rejects_negative_gpu_id():
    server = make_server()

    try:
        server.start_keep(gpu_ids=[0, -1])
    except ValueError as exc:
        assert "gpu_ids must contain non-negative integers" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert server.status()["active_jobs"] == []


def test_start_rejects_empty_gpu_ids():
    server = make_server()

    try:
        server.start_keep(gpu_ids=[])
    except ValueError as exc:
        assert "gpu_ids must select at least one GPU" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert server.status()["active_jobs"] == []


def test_start_rejects_duplicate_gpu_ids():
    server = make_server()

    try:
        server.start_keep(gpu_ids=[0, 1, 0])
    except ValueError as exc:
        assert "gpu_ids must not contain duplicate values" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_empty_gpu_ids():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": []},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "gpu_ids must select at least one GPU" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_duplicate_gpu_ids():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0, 1, 0]},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "gpu_ids must not contain duplicate values" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_non_positive_interval():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "interval": 0},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "interval must be positive" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_nan_interval_without_creating_session():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "interval": math.nan},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "interval must be finite and positive" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_start_keep_preserves_fractional_interval():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "interval": 0.5},
    }

    resp = _handle_request(server, req)

    assert resp["result"]["job_id"]
    status = server.status(resp["result"]["job_id"])
    assert status["params"]["interval"] == 0.5


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"gpu_ids": [0], "interval": 10**1000}, "interval must be no more than"),
        ({"gpu_ids": [0], "vram": 10**1000}, "vram must be no more than"),
        (
            {"gpu_ids": [0], "vram": ("9" * 500) + "GiB"},
            "vram must be no more than",
        ),
    ],
)
def test_jsonrpc_rejects_oversized_numeric_session_inputs_without_internal_error(
    params, message
):
    server = make_server()
    req = {"id": 1, "method": "start_keep", "params": params}

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert message in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_busy_threshold_above_percent_range():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "interval": 1, "busy_threshold": 101},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert (
        "busy_threshold must be -1 or an integer between 0 and 100"
        in resp["error"]["message"]
    )
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_invalid_vram_type_without_creating_session():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "vram": []},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "vram_to_keep must be str or int bytes" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_rejects_unknown_direct_method_param_without_internal_error():
    server = make_server()
    req = {
        "id": 1,
        "method": "status",
        "params": {"unexpected": True},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "Unknown params for status" in resp["error"]["message"]


def test_jsonrpc_start_keep_runtime_value_error_remains_internal_error():
    def failing_factory(**kwargs):
        raise ValueError("controller startup failed")

    server = KeepGPUServer(controller_factory=cast(Any, failing_factory))
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0]},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INTERNAL_ERROR
    assert "controller startup failed" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_cuda_worker_startup_failure_creates_no_active_session(monkeypatch):
    import torch

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fail_set_device(_rank):
        raise RuntimeError("cuda worker startup failed")

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", fail_set_device)

    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "startup-fails", "gpu_ids": [0]},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_INTERNAL_ERROR
        assert "cuda worker startup failed" in resp["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_jsonrpc_start_keep_defaults_to_eco_safe_busy_threshold():
    server = make_server()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "jsonrpc-default", "gpu_ids": [0]},
    }

    resp = _handle_request(server, req)

    assert resp["result"] == {"job_id": "jsonrpc-default"}
    status = server.status("jsonrpc-default")
    assert status["params"]["busy_threshold"] == 25


@pytest.mark.parametrize("job_id", ["", " ", 123, "job/123", "job?123", "job#123"])
def test_start_keep_rejects_invalid_job_id_before_starting_controller(job_id):
    controllers = []

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            controllers.append(self)
            super().__init__(**kwargs)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )

    with pytest.raises(ValueError, match="job_id"):
        server.start_keep(job_id=job_id)

    assert controllers == []
    assert server.status()["active_jobs"] == []


@pytest.mark.parametrize("job_id", ["", " ", 123, "job/123", "job?123", "job#123"])
def test_status_rejects_invalid_job_id_without_changing_sessions(job_id):
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]

    with pytest.raises(ValueError, match="job_id"):
        server.status(job_id=job_id)

    assert server.status(active_job_id)["active"] is True


@pytest.mark.parametrize("job_id", ["", " ", 123, "job/123", "job?123", "job#123"])
def test_stop_keep_rejects_invalid_job_id_without_stopping_sessions(job_id):
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller

    with pytest.raises(ValueError, match="job_id"):
        server.stop_keep(job_id=job_id)

    assert server.status(active_job_id)["active"] is True
    assert controller.released is False


def test_jsonrpc_stop_keep_rejects_empty_job_id_without_stopping_sessions():
    server = make_server()
    active_job_id = server.start_keep(job_id="active-job")["job_id"]
    controller = server._sessions[active_job_id].controller
    req = {"id": 1, "method": "stop_keep", "params": {"job_id": ""}}

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
    assert "job_id" in resp["error"]["message"]
    assert server.status(active_job_id)["active"] is True
    assert controller.released is False


def test_stop_all():
    server = make_server()
    job_a = server.start_keep()["job_id"]
    job_b = server.start_keep()["job_id"]

    stopped = server.stop_keep()
    assert set(stopped["stopped"]) == {job_a, job_b}
    assert server.status(job_a)["active"] is False
    assert server.status(job_b)["active"] is False


def test_list_gpus():
    server = make_server()
    info = server.list_gpus()
    assert "gpus" in info


def test_mcp_initialize_returns_server_capabilities():
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0"},
        },
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    result = resp["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "keepgpu"
    assert result["serverInfo"]["title"] == "KeepGPU"
    assert result["serverInfo"]["version"]


def test_mcp_initialized_notification_has_no_response():
    server = make_server()
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    resp = _handle_request(server, req)

    assert resp is None


def test_mcp_tools_list_exposes_keepgpu_actions():
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 2
    tools = {tool["name"]: tool for tool in resp["result"]["tools"]}
    assert set(tools) == {"start_keep", "stop_keep", "status", "list_gpus"}
    start_schema = tools["start_keep"]["inputSchema"]
    assert start_schema["type"] == "object"
    assert start_schema["properties"]["gpu_ids"]["items"]["type"] == "integer"
    assert set(start_schema["properties"]["vram"]["type"]) == {"string", "integer"}
    assert start_schema["properties"]["vram"]["maximum"] == PUBLIC_VRAM_MAX_BYTES
    assert "1 PiB" in start_schema["properties"]["vram"]["description"]
    assert start_schema["properties"]["interval"]["type"] == "number"
    assert start_schema["properties"]["interval"]["exclusiveMinimum"] == 0
    assert start_schema["properties"]["busy_threshold"]["default"] == 25
    assert tools["status"]["inputSchema"]["properties"]["job_id"]["type"] == [
        "string",
        "null",
    ]


def test_mcp_tools_call_routes_to_existing_status_method():
    server = make_server()
    job_id = server.start_keep(job_id="mcp-job", gpu_ids=[0])["job_id"]
    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "status", "arguments": {"job_id": job_id}},
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 3
    result = resp["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    payload = json.loads(result["content"][0]["text"])
    assert payload["active"] is True
    assert payload["job_id"] == "mcp-job"
    assert payload["params"]["gpu_ids"] == [0]


def test_mcp_tools_call_rejects_oversized_integer_vram_as_tool_error():
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "start_keep",
            "arguments": {"gpu_ids": [0], "vram": 10**1000},
        },
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 4
    assert "result" in resp
    result = resp["result"]
    assert result["isError"] is True
    assert "vram must be no more than" in result["content"][0]["text"]
    assert server.status()["active_jobs"] == []


def test_mcp_tools_call_unknown_tool_returns_protocol_error():
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "not_a_tool", "arguments": {}},
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 4
    assert resp["error"]["code"] == -32602
    assert resp["error"]["message"] == "Unknown tool: not_a_tool"


def test_mcp_tools_call_rejects_non_object_arguments():
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "status", "arguments": []},
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 5
    assert resp["error"]["code"] == -32602
    assert resp["error"]["message"] == "Tool call arguments must be an object."


def test_jsonrpc_unknown_method_returns_method_not_found_code():
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 6, "method": "not_a_method", "params": {}}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 6
    assert resp["error"]["code"] == -32601
    assert resp["error"]["message"] == "Unknown method: not_a_method"


def test_mcp_requests_require_id():
    server = make_server()
    req = {"jsonrpc": "2.0", "method": "tools/list", "params": {}}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] is None
    assert resp["error"]["code"] == -32600
    assert resp["error"]["message"] == "Requests must include an id."


def test_mcp_unrecognized_notification_has_no_response():
    server = make_server()
    req = {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}}

    resp = _handle_request(server, req)

    assert resp is None


def test_mcp_notification_with_id_is_invalid_request():
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 7, "method": "notifications/initialized"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 7
    assert resp["error"]["code"] == -32600
    assert resp["error"]["message"] == "Notifications must not include an id."


def test_mcp_stdio_stdout_contains_only_protocol_json():
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/list",
    }
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, "-m", "keep_gpu.mcp.server"],
        input=json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=5,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 8
    assert sorted(tool["name"] for tool in response["result"]["tools"]) == [
        "list_gpus",
        "start_keep",
        "status",
        "stop_keep",
    ]


def test_mcp_stdio_parse_errors_are_jsonrpc_errors():
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, "-m", "keep_gpu.mcp.server"],
        input="{not json}\n",
        text=True,
        capture_output=True,
        timeout=5,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] is None
    assert response["error"]["code"] == -32700
    assert "Expecting property name" in response["error"]["message"]


def test_mcp_stdio_non_object_messages_are_invalid_request_errors():
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, "-m", "keep_gpu.mcp.server"],
        input='["not", "an", "object"]\n',
        text=True,
        capture_output=True,
        timeout=5,
        env=env,
        check=False,
    )

    assert completed.returncode == 0
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(stdout_lines) == 1
    response = json.loads(stdout_lines[0])
    assert response["jsonrpc"] == "2.0"
    assert response["id"] is None
    assert response["error"]["code"] == -32600
    assert response["error"]["message"] == "JSON-RPC messages must be objects."


def test_end_to_end_jsonrpc():
    server = make_server()
    # start_keep
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"gpu_ids": [0], "vram": "256MB", "interval": 1, "busy_threshold": 5},
    }
    resp = _handle_request(server, req)
    assert "result" in resp and "job_id" in resp["result"]
    job_id = resp["result"]["job_id"]

    # status
    status_req = {"id": 2, "method": "status", "params": {"job_id": job_id}}
    status_resp = _handle_request(server, status_req)
    assert status_resp["result"]["active"] is True

    # stop_keep
    stop_req = {"id": 3, "method": "stop_keep", "params": {"job_id": job_id}}
    stop_resp = _handle_request(server, stop_req)
    assert job_id in stop_resp["result"]["stopped"]


def test_status_all():
    server = make_server()
    job_a = server.start_keep(gpu_ids=[0])["job_id"]
    job_b = server.start_keep(gpu_ids=[1])["job_id"]

    status = server.status()
    assert "active_jobs" in status
    assert len(status["active_jobs"]) == 2

    job_statuses = {job["job_id"]: job for job in status["active_jobs"]}
    assert job_a in job_statuses
    assert job_b in job_statuses
    assert job_statuses[job_a]["params"]["gpu_ids"] == [0]
    assert job_statuses[job_b]["params"]["gpu_ids"] == [1]
    assert "controller" not in job_statuses[job_a]


def test_concurrent_duplicate_job_id_rejected_before_second_keep():
    first_keep_entered = threading.Event()
    first_keep_release = threading.Event()
    controllers = []
    keep_calls = []
    first_result = {}

    class SlowFirstController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.index = len(controllers) + 1
            controllers.append(self)

        def keep(self):
            keep_calls.append(self.index)
            self.kept = True
            if self.index == 1:
                first_keep_entered.set()
                first_keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowFirstController(**kwargs))
    )

    def start_first():
        try:
            first_result["value"] = server.start_keep(job_id="shared-job")
        except Exception as exc:  # pragma: no cover - failure diagnostic
            first_result["error"] = exc

    thread = threading.Thread(target=start_first)
    thread.start()
    assert first_keep_entered.wait(timeout=1.0)

    second_error = None
    try:
        try:
            server.start_keep(job_id="shared-job")
        except ValueError as exc:
            second_error = exc
    finally:
        first_keep_release.set()
        thread.join(timeout=1.0)

    assert isinstance(second_error, ValueError)
    assert "job_id shared-job already exists" in str(second_error)
    assert len(controllers) == 1
    assert keep_calls == [1]
    assert first_result["value"] == {"job_id": "shared-job"}
    assert server.status("shared-job")["active"] is True


def test_failed_start_releases_job_id_reservation():
    attempts = 0

    class FailsOnceController(DummyController):
        def keep(self):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("start failed")
            self.kept = True

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: FailsOnceController(**kwargs))
    )

    try:
        server.start_keep(job_id="retry-job")
    except RuntimeError as exc:
        assert "start failed" in str(exc)
    else:
        raise AssertionError("Expected first start to fail")

    result = server.start_keep(job_id="retry-job")

    assert result == {"job_id": "retry-job"}
    assert attempts == 2
    assert server.status("retry-job")["active"] is True


def test_factory_failure_releases_job_id_reservation():
    attempts = 0

    def factory(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("factory failed")
        return DummyController(**kwargs)

    server = KeepGPUServer(controller_factory=cast(Any, factory))

    try:
        server.start_keep(job_id="factory-retry-job")
    except RuntimeError as exc:
        assert "factory failed" in str(exc)
    else:
        raise AssertionError("Expected first start to fail")

    result = server.start_keep(job_id="factory-retry-job")

    assert result == {"job_id": "factory-retry-job"}
    assert attempts == 2
    assert server.status("factory-retry-job")["active"] is True


def test_stop_keep_waits_for_starting_session(monkeypatch):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    stop_waiting_for_startup = threading.Event()
    controllers = []
    start_result = {}
    stop_result = {}

    class SlowStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowStartController(**kwargs))
    )
    original_wait = server._sessions_cond.wait

    def wait_for_startup(timeout=None):
        stop_waiting_for_startup.set()
        return original_wait(timeout)

    monkeypatch.setattr(server._sessions_cond, "wait", wait_for_startup)

    def start_job():
        start_result["value"] = server.start_keep(job_id="starting-job")

    def stop_job():
        stop_result["value"] = server.stop_keep("starting-job")

    start_thread = threading.Thread(target=start_job)
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(target=stop_job)
    stop_thread.start()
    assert stop_waiting_for_startup.wait(timeout=1.0)
    keep_release.set()

    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert start_result["value"] == {"job_id": "starting-job"}
    assert stop_result["value"]["stopped"] == ["starting-job"]
    assert controllers[0].released is True
    assert server.status("starting-job")["active"] is False


def test_stop_all_waits_for_starting_session(monkeypatch):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    stop_waiting_for_startup = threading.Event()
    controllers = []
    start_result = {}
    stop_result = {}

    class SlowStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowStartController(**kwargs))
    )
    original_wait = server._sessions_cond.wait

    def wait_for_startup(timeout=None):
        stop_waiting_for_startup.set()
        return original_wait(timeout)

    monkeypatch.setattr(server._sessions_cond, "wait", wait_for_startup)

    def start_job():
        start_result["value"] = server.start_keep(job_id="starting-job")

    def stop_all():
        stop_result["value"] = server.stop_keep()

    start_thread = threading.Thread(target=start_job)
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(target=stop_all)
    stop_thread.start()
    assert stop_waiting_for_startup.wait(timeout=1.0)
    keep_release.set()

    start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert start_result["value"] == {"job_id": "starting-job"}
    assert stop_result["value"]["stopped"] == ["starting-job"]
    assert controllers[0].released is True
    assert server.status("starting-job")["active"] is False


def test_stop_all_waits_only_for_sessions_starting_at_snapshot(monkeypatch):
    keep_entered = [threading.Event(), threading.Event()]
    keep_release = [threading.Event(), threading.Event()]
    stop_waiting_for_startup = threading.Event()
    stop_completed = threading.Event()
    controllers = []
    start_results = {}
    stop_result = {}

    class SlowStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.index = len(controllers)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered[self.index].set()
            keep_release[self.index].wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowStartController(**kwargs))
    )
    original_wait = server._sessions_cond.wait

    def wait_for_startup(timeout=None):
        stop_waiting_for_startup.set()
        return original_wait(timeout)

    monkeypatch.setattr(server._sessions_cond, "wait", wait_for_startup)

    def start_job(job_id):
        start_results[job_id] = server.start_keep(job_id=job_id)

    def stop_all():
        stop_result["value"] = server.stop_keep()
        stop_completed.set()

    first_start_thread = threading.Thread(target=start_job, args=("first-job",))
    first_start_thread.start()
    assert keep_entered[0].wait(timeout=1.0)

    stop_thread = threading.Thread(target=stop_all)
    stop_thread.start()
    assert stop_waiting_for_startup.wait(timeout=1.0)

    second_start_thread = threading.Thread(target=start_job, args=("second-job",))
    second_start_thread.start()
    assert keep_entered[1].wait(timeout=1.0)

    keep_release[0].set()
    first_start_thread.join(timeout=1.0)
    assert stop_completed.wait(timeout=1.0)

    assert start_results["first-job"] == {"job_id": "first-job"}
    assert stop_result["value"]["stopped"] == ["first-job"]
    assert controllers[0].released is True
    assert controllers[1].released is False

    keep_release[1].set()
    second_start_thread.join(timeout=1.0)
    stop_thread.join(timeout=1.0)

    assert start_results["second-job"] == {"job_id": "second-job"}
    assert server.status("first-job")["active"] is False
    assert server.status("second-job")["active"] is True
    server.stop_keep("second-job")


def test_stop_keep_returns_timeout_payload(monkeypatch):
    server = make_server()
    job_id = server.start_keep()["job_id"]

    monkeypatch.setattr(server, "_release_with_timeout", lambda controller, **_: False)

    result = server.stop_keep(job_id)
    assert result["stopped"] == []
    assert result["timed_out"] == [job_id]
    assert "Timed out" in result["message"]
    status = server.status(job_id)
    assert status["active"] is True
    assert status["state"] == "stopping"
    assert "Timed out" in status["last_error"]


def test_stop_keep_returns_failed_payload_and_retains_session(monkeypatch):
    server = make_server()
    job_id = server.start_keep()["job_id"]

    def fail_release(controller, **_):
        raise RuntimeError("release exploded")

    monkeypatch.setattr(server, "_release_with_timeout", fail_release)

    result = server.stop_keep(job_id)
    assert result["stopped"] == []
    assert result["failed"] == [job_id]
    assert result["errors"] == {job_id: "release exploded"}
    status = server.status(job_id)
    assert status["active"] is True
    assert status["state"] == "stop_failed"
    assert status["last_error"] == "release exploded"


def test_stop_all_tracks_timeouts(monkeypatch):
    server = make_server()
    job_a = server.start_keep()["job_id"]
    job_b = server.start_keep()["job_id"]

    def release_outcome(controller, **_):
        return controller is not server._sessions[job_b].controller

    monkeypatch.setattr(server, "_release_with_timeout", release_outcome)

    result = server.stop_keep()
    assert result["stopped"] == [job_a]
    assert result["timed_out"] == [job_b]
    assert server.status(job_a)["active"] is False
    status_b = server.status(job_b)
    assert status_b["active"] is True
    assert status_b["state"] == "stopping"


def test_stop_all_orders_new_timeouts_before_later_already_stopping(monkeypatch):
    server = make_server()
    job_timeout = server.start_keep(gpu_ids=[0])["job_id"]
    job_stopping = server.start_keep(gpu_ids=[1])["job_id"]

    monkeypatch.setattr(server, "_release_with_timeout", lambda controller, **_: False)
    targeted_result = server.stop_keep(job_stopping)
    assert targeted_result["timed_out"] == [job_stopping]

    stop_all_controllers = []

    def timeout_release(controller, **_):
        stop_all_controllers.append(controller)
        return False

    monkeypatch.setattr(server, "_release_with_timeout", timeout_release)

    result = server.stop_keep()

    assert result["timed_out"] == [job_timeout, job_stopping]
    assert [controller.gpu_ids for controller in stop_all_controllers] == [[0]]


def test_stop_all_release_workers_enter_concurrently(monkeypatch):
    server = make_server()
    job_ids = [
        server.start_keep(gpu_ids=[0])["job_id"],
        server.start_keep(gpu_ids=[1])["job_id"],
        server.start_keep(gpu_ids=[2])["job_id"],
    ]
    entered_count = 0
    entered_lock = threading.Lock()
    all_entered = threading.Event()
    release_gate = threading.Event()
    stop_result = {}

    def blocking_release(controller, **_):
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
            if entered_count == len(job_ids):
                all_entered.set()
        release_gate.wait(timeout=1.0)
        return True

    monkeypatch.setattr(server, "_release_with_timeout", blocking_release)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    stop_thread.start()

    try:
        assert all_entered.wait(timeout=2.0)
    finally:
        release_gate.set()
        stop_thread.join(timeout=1.0)

    assert stop_result["value"]["stopped"] == job_ids
    assert all(server.status(job_id)["active"] is False for job_id in job_ids)


def test_stop_all_concurrent_results_keep_snapshot_order(monkeypatch):
    server = make_server()
    job_success = server.start_keep(gpu_ids=[0])["job_id"]
    job_timeout = server.start_keep(gpu_ids=[1])["job_id"]
    job_failed = server.start_keep(gpu_ids=[2])["job_id"]
    job_ids = [job_success, job_timeout, job_failed]
    entered_count = 0
    entered_lock = threading.Lock()
    all_entered = threading.Event()
    release_gate = threading.Event()
    stop_result = {}

    def release_outcome(controller, **_):
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
            if entered_count == len(job_ids):
                all_entered.set()
        release_gate.wait(timeout=1.0)
        if controller.gpu_ids == [1]:
            return False
        if controller.gpu_ids == [2]:
            raise RuntimeError("release failed")
        return True

    monkeypatch.setattr(server, "_release_with_timeout", release_outcome)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    stop_thread.start()

    try:
        assert all_entered.wait(timeout=2.0)
    finally:
        release_gate.set()
        stop_thread.join(timeout=1.0)

    assert stop_result["value"]["stopped"] == [job_success]
    assert stop_result["value"]["timed_out"] == [job_timeout]
    assert stop_result["value"]["failed"] == [job_failed]
    assert stop_result["value"]["errors"] == {job_failed: "release failed"}
    assert server.status(job_success)["active"] is False
    assert server.status(job_timeout)["state"] == "stopping"
    assert server.status(job_failed)["state"] == "stop_failed"


def test_timed_out_stop_removes_session_after_background_release_succeeds(monkeypatch):
    release_gate = threading.Event()

    class SlowSuccessController(DummyController):
        def release(self):
            release_gate.wait(timeout=1.0)
            self.released = True

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowSuccessController(**kwargs))
    )
    original_release_with_timeout = server._release_with_timeout

    def short_timeout(controller, **kwargs):
        kwargs["timeout_s"] = 0.01
        return original_release_with_timeout(controller, **kwargs)

    monkeypatch.setattr(server, "_release_with_timeout", short_timeout)
    job_id = server.start_keep()["job_id"]

    result = server.stop_keep(job_id)

    assert result["timed_out"] == [job_id]
    assert server.status(job_id)["state"] == "stopping"

    release_gate.set()
    assert _wait_until(lambda: server.status(job_id)["active"] is False)


def test_timed_out_stop_marks_late_background_release_failure(monkeypatch):
    release_gate = threading.Event()

    class SlowFailController(DummyController):
        def release(self):
            release_gate.wait(timeout=1.0)
            raise RuntimeError("late release failed")

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowFailController(**kwargs))
    )
    original_release_with_timeout = server._release_with_timeout

    def short_timeout(controller, **kwargs):
        kwargs["timeout_s"] = 0.01
        return original_release_with_timeout(controller, **kwargs)

    monkeypatch.setattr(server, "_release_with_timeout", short_timeout)
    job_id = server.start_keep()["job_id"]

    result = server.stop_keep(job_id)

    assert result["timed_out"] == [job_id]
    release_gate.set()
    assert _wait_until(lambda: server.status(job_id).get("state") == "stop_failed")
    status = server.status(job_id)
    assert status["active"] is True
    assert status["last_error"] == "late release failed"


def test_stop_all_late_callbacks_update_each_timed_out_session(monkeypatch):
    server = make_server()
    job_late_success = server.start_keep(gpu_ids=[0])["job_id"]
    job_late_failure = server.start_keep(gpu_ids=[1])["job_id"]

    def timeout_with_late_callback(controller, on_late_result, **_):
        if controller.gpu_ids == [0]:
            on_late_result(None)
        else:
            on_late_result(RuntimeError("late release failed"))
        return False

    monkeypatch.setattr(server, "_release_with_timeout", timeout_with_late_callback)

    result = server.stop_keep()

    assert result["timed_out"] == [job_late_success, job_late_failure]
    assert server.status(job_late_success)["active"] is False
    status = server.status(job_late_failure)
    assert status["active"] is True
    assert status["state"] == "stop_failed"
    assert status["last_error"] == "late release failed"


def test_timed_out_stop_preserves_failure_from_timeout_race(monkeypatch):
    server = make_server()
    job_id = server.start_keep()["job_id"]

    def timeout_after_late_failure(controller, **kwargs):
        kwargs["on_late_result"](RuntimeError("late release failed"))
        return False

    monkeypatch.setattr(server, "_release_with_timeout", timeout_after_late_failure)

    result = server.stop_keep(job_id)

    assert result["timed_out"] == [job_id]
    status = server.status(job_id)
    assert status["active"] is True
    assert status["state"] == "stop_failed"
    assert status["last_error"] == "late release failed"


def test_repeated_stop_does_not_start_second_release_while_stopping(monkeypatch):
    release_gate = threading.Event()
    release_calls = 0

    class SlowController(DummyController):
        def release(self):
            nonlocal release_calls
            release_calls += 1
            release_gate.wait(timeout=1.0)
            self.released = True

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowController(**kwargs))
    )
    original_release_with_timeout = server._release_with_timeout

    def short_timeout(controller, **kwargs):
        kwargs["timeout_s"] = 0.01
        return original_release_with_timeout(controller, **kwargs)

    monkeypatch.setattr(server, "_release_with_timeout", short_timeout)
    job_id = server.start_keep()["job_id"]

    first = server.stop_keep(job_id)
    second = server.stop_keep(job_id)

    assert first["timed_out"] == [job_id]
    assert second["timed_out"] == [job_id]
    assert release_calls == 1

    release_gate.set()
    assert _wait_until(lambda: server.status(job_id)["active"] is False)


def test_stop_all_does_not_restart_already_stopping_session(monkeypatch):
    release_gate = threading.Event()
    release_calls = {}

    class SlowController(DummyController):
        def release(self):
            key = self.gpu_ids[0]
            release_calls[key] = release_calls.get(key, 0) + 1
            if key == 0:
                release_gate.wait(timeout=1.0)
            self.released = True

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: SlowController(**kwargs))
    )
    original_release_with_timeout = server._release_with_timeout

    def short_timeout(controller, **kwargs):
        if controller.gpu_ids == [0]:
            kwargs["timeout_s"] = 0.01
        return original_release_with_timeout(controller, **kwargs)

    monkeypatch.setattr(server, "_release_with_timeout", short_timeout)
    job_a = server.start_keep(gpu_ids=[0])["job_id"]
    job_b = server.start_keep(gpu_ids=[1])["job_id"]

    first = server.stop_keep(job_a)
    second = server.stop_keep()

    assert first["timed_out"] == [job_a]
    assert second["timed_out"] == [job_a]
    assert second["stopped"] == [job_b]
    assert release_calls == {0: 1, 1: 1}

    release_gate.set()
    assert _wait_until(lambda: server.status(job_a)["active"] is False)
    assert server.status(job_b)["active"] is False


def test_stop_all_reports_failures_and_continues(monkeypatch):
    server = make_server()
    job_a = server.start_keep(gpu_ids=[0])["job_id"]
    job_b = server.start_keep(gpu_ids=[1])["job_id"]
    job_c = server.start_keep(gpu_ids=[2])["job_id"]

    def release_outcome(controller, **_):
        if controller.gpu_ids == [1]:
            raise RuntimeError("release failed")
        return True

    monkeypatch.setattr(server, "_release_with_timeout", release_outcome)

    result = server.stop_keep()
    assert result["stopped"] == [job_a, job_c]
    assert result["failed"] == [job_b]
    assert result["errors"] == {job_b: "release failed"}
    assert server.status(job_a)["active"] is False
    assert server.status(job_c)["active"] is False
    status_b = server.status(job_b)
    assert status_b["active"] is True
    assert status_b["state"] == "stop_failed"
