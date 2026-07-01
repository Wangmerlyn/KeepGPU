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

from keep_gpu.mcp import server as server_module
from keep_gpu.mcp.server import (
    JSONRPC_INTERNAL_ERROR,
    JSONRPC_INVALID_PARAMS,
    JSONRPC_INVALID_REQUEST,
    JSONRPC_STARTUP_UNAVAILABLE,
    KeepGPUServer,
    SessionStartupUnavailable,
    _handle_request,
)
from keep_gpu.utilities import gpu_info, platform_manager as pm
from keep_gpu.utilities.humanized_input import PUBLIC_VRAM_MAX_BYTES
from keep_gpu.utilities.session_config import JOB_ID_PATTERN_TEXT


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


@pytest.mark.parametrize(
    ("endpoint_args", "expected_error"),
    [
        (["--port", "0"], "port must be an integer between 1 and 65535"),
        (["--port", "70000"], "port must be an integer between 1 and 65535"),
        (["--port", "true"], "port must be an integer between 1 and 65535"),
        (["--port", "+8765"], "port must be an integer between 1 and 65535"),
        (["--port", "8_765"], "port must be an integer between 1 and 65535"),
        (["--port", "１２３"], "port must be an integer between 1 and 65535"),
        (["--host", "bad host"], "host must be a DNS hostname or IPv4 address"),
    ],
)
def test_http_main_rejects_invalid_endpoint_before_binding(
    monkeypatch, capsys, endpoint_args, expected_error
):
    run_http_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        ["keep-gpu-mcp-server", "--mode", "http", *endpoint_args],
    )
    monkeypatch.setattr(server_module, "KeepGPUServer", lambda: object())
    monkeypatch.setattr(
        server_module,
        "run_http",
        lambda *args, **kwargs: run_http_calls.append((args, kwargs)),
    )

    with pytest.raises(SystemExit) as exc_info:
        server_module.main()

    assert exc_info.value.code != 0
    assert run_http_calls == []
    assert expected_error in capsys.readouterr().err


def test_http_main_passes_valid_endpoint_to_run_http(monkeypatch):
    run_http_calls = []
    server_instance = object()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "keep-gpu-mcp-server",
            "--mode",
            "http",
            "--host",
            "localhost",
            "--port",
            "9876",
        ],
    )
    monkeypatch.setattr(server_module, "KeepGPUServer", lambda: server_instance)
    monkeypatch.setattr(
        server_module,
        "run_http",
        lambda *args, **kwargs: run_http_calls.append((args, kwargs)),
    )

    server_module.main()

    assert run_http_calls == [((server_instance,), {"host": "localhost", "port": 9876})]


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


def test_status_returns_param_snapshots_for_active_sessions():
    server = make_server()
    job_id = server.start_keep(job_id="snapshot-job", gpu_ids=[0])["job_id"]

    single_status = server.status(job_id)
    single_status["params"]["gpu_ids"].append(99)
    single_status["params"]["vram"] = "8GiB"

    all_status = server.status()
    all_status["active_jobs"][0]["params"]["gpu_ids"].append(42)

    assert server.status(job_id)["params"] == {
        "gpu_ids": [0],
        "vram": "1GiB",
        "interval": 300,
        "busy_threshold": 25,
    }


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


@pytest.mark.parametrize("status_scope", ["single", "all"])
def test_status_runtime_health_hook_does_not_block_stop_keep(status_scope):
    runtime_hook_entered = threading.Event()
    release_runtime_hook = threading.Event()
    status_result = {}
    stop_result = {}

    class BlockingRuntimeHealthController(DummyController):
        def runtime_error(self):
            runtime_hook_entered.set()
            release_runtime_hook.wait(timeout=2.0)
            return RuntimeError("late runtime failure")

    server = KeepGPUServer(
        controller_factory=cast(
            Any, lambda **kwargs: BlockingRuntimeHealthController(**kwargs)
        )
    )
    job_id = server.start_keep(job_id="slow-health", gpu_ids=[0])["job_id"]

    def check_status():
        if status_scope == "single":
            status_result.update(value=server.status(job_id))
            return
        status_result.update(value=server.status())

    status_thread = threading.Thread(target=check_status)
    status_thread.start()
    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep(job_id))
    )
    try:
        assert runtime_hook_entered.wait(timeout=1.0)

        stop_thread.start()

        assert _wait_until(lambda: "value" in stop_result, timeout_s=0.2)
        assert stop_result["value"]["stopped"] == [job_id]
        assert server._sessions == {}
    finally:
        release_runtime_hook.set()
        stop_thread.join(timeout=1.0)
        status_thread.join(timeout=1.0)

    assert not stop_thread.is_alive()
    assert not status_thread.is_alive()
    if status_scope == "single":
        assert status_result["value"] == {"active": False, "job_id": job_id}
    else:
        assert status_result["value"] == {"active_jobs": []}


def test_late_runtime_health_result_does_not_mutate_reused_job_id():
    runtime_hook_entered = threading.Event()
    release_runtime_hook = threading.Event()
    status_result = {}
    controller_count = 0

    class BlockingRuntimeHealthController(DummyController):
        def __init__(self, **kwargs):
            nonlocal controller_count

            super().__init__(**kwargs)
            controller_count += 1
            self.should_block_runtime = controller_count == 1

        def runtime_error(self):
            if self.should_block_runtime:
                runtime_hook_entered.set()
                release_runtime_hook.wait(timeout=2.0)
                return RuntimeError("stale runtime failure")
            return None

    server = KeepGPUServer(
        controller_factory=cast(
            Any, lambda **kwargs: BlockingRuntimeHealthController(**kwargs)
        )
    )
    job_id = server.start_keep(job_id="reused-job", gpu_ids=[0])["job_id"]
    status_thread = threading.Thread(
        target=lambda: status_result.update(value=server.status(job_id))
    )
    status_thread.start()
    try:
        assert runtime_hook_entered.wait(timeout=1.0)
        assert server.stop_keep(job_id)["stopped"] == [job_id]

        assert server.start_keep(job_id=job_id, gpu_ids=[1]) == {"job_id": job_id}

        release_runtime_hook.set()
        status_thread.join(timeout=1.0)
    finally:
        release_runtime_hook.set()
        status_thread.join(timeout=1.0)

    assert not status_thread.is_alive()
    assert status_result["value"]["state"] == "active"
    assert status_result["value"]["last_error"] is None
    assert status_result["value"]["params"]["gpu_ids"] == [1]
    assert server.status(job_id)["state"] == "active"


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
        single_status = server.status("starting-job")
        single_status["params"]["gpu_ids"].append(99)
        list_status = server.status()
        list_status["active_jobs"][0]["params"]["gpu_ids"].append(42)
        assert server.status("starting-job")["params"] == expected_params
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


def test_jsonrpc_start_keep_startup_unavailable_returns_public_code():
    def failing_factory(**kwargs):
        raise SessionStartupUnavailable("No usable GPUs are available")

    server = KeepGPUServer(controller_factory=cast(Any, failing_factory))
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "no-usable-gpus", "gpu_ids": [0]},
    }

    resp = _handle_request(server, req)

    assert "error" in resp
    assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
    assert "No usable GPUs are available" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_jsonrpc_start_keep_unsupported_platform_returns_public_code(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CPU)
    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "unsupported-platform"},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
        assert (
            "GlobalGPUController not implemented for platform"
            in resp["error"]["message"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_jsonrpc_start_keep_unavailable_mps_returns_public_code(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "mps-unavailable", "gpu_ids": [0]},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
        assert "PyTorch MPS backend is not available" in resp["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_jsonrpc_start_keep_mps_probe_exception_returns_public_code(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    def raise_probe_error():
        raise ValueError("MPS probe exploded")

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        raise_probe_error,
    )

    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "mps-probe-error", "gpu_ids": [0]},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
        assert (
            "PyTorch MPS backend availability check failed: MPS probe exploded"
            in resp["error"]["message"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_jsonrpc_start_keep_zero_visible_gpus_returns_public_code(monkeypatch):
    import torch

    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "zero-visible-gpus"},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
        assert "No GPUs available for GlobalGPUController" in resp["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("platform", "device_counts", "job_id", "message"),
    [
        (
            pm.ComputingPlatform.CUDA,
            [RuntimeError("cuda runtime unavailable")],
            "device-count-failed",
            "cuda runtime unavailable",
        ),
        (
            pm.ComputingPlatform.CUDA,
            [1, RuntimeError("cuda runtime disappeared")],
            "late-device-count-failed",
            "cuda runtime disappeared",
        ),
        (
            pm.ComputingPlatform.CUDA,
            [1, 0],
            "late-zero-device-count",
            "no visible GPUs are available",
        ),
        (
            pm.ComputingPlatform.ROCM,
            [1, 0],
            "rocm-late-zero-device-count",
            "no visible GPUs are available",
        ),
    ],
)
def test_jsonrpc_start_keep_device_enumeration_unavailable_returns_startup_unavailable(
    monkeypatch, platform, device_counts, job_id, message
):
    import torch

    monkeypatch.setattr(pm, "_cached_platform", platform)
    calls = iter(device_counts)

    def device_count():
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": job_id},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
        assert message in resp["error"]["message"]
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_jsonrpc_start_keep_out_of_range_gpu_ids_returns_invalid_params(monkeypatch):
    import torch

    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    server = KeepGPUServer()
    req = {
        "id": 1,
        "method": "start_keep",
        "params": {"job_id": "bad-visible-gpu", "gpu_ids": [99]},
    }

    try:
        resp = _handle_request(server, req)

        assert "error" in resp
        assert resp["error"]["code"] == JSONRPC_INVALID_PARAMS
        assert (
            "gpu_ids must be visible device ordinals less than 1"
            in resp["error"]["message"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


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


def test_list_gpus_propagates_real_helper_enumeration_unavailable(
    monkeypatch,
):
    class DummyNVML:
        shutdown_calls = 0

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 1

        @classmethod
        def nvmlShutdown(cls):
            cls.shutdown_calls += 1

    class DummyCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            raise RuntimeError("cuda runtime unavailable")

    dummy_torch = type(
        "T",
        (),
        {
            "cuda": DummyCuda(),
            "version": type("V", (), {"hip": None}),
            "backends": type(
                "Backends",
                (),
                {
                    "mps": type(
                        "MPSBackend", (), {"is_available": staticmethod(lambda: False)}
                    )
                },
            ),
        },
    )
    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    server = make_server()

    with pytest.raises(
        pm.DeviceEnumerationUnavailableError,
        match="Unable to enumerate visible GPUs: cuda runtime unavailable",
    ):
        server.list_gpus()
    assert DummyNVML.shutdown_calls == 1


def test_list_gpus_accepts_nullable_memory_and_utilization(monkeypatch):
    records = [
        {
            "id": 0,
            "visible_id": 0,
            "platform": "CUDA",
            "name": "GPU 0",
            "memory_total": None,
            "memory_used": None,
            "utilization": None,
        }
    ]
    monkeypatch.setattr(server_module, "get_gpu_info", lambda: records)
    server = make_server()

    assert server.list_gpus() == {"gpus": records}


@pytest.mark.parametrize(
    ("records", "message_fragment"),
    [
        (
            [
                {
                    "id": -1,
                    "visible_id": -1,
                    "platform": "CUDA",
                    "name": "GPU hidden",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                }
            ],
            "non-negative",
        ),
        (
            [
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU 0",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                },
                {
                    "id": 0,
                    "visible_id": 0,
                    "platform": "CUDA",
                    "name": "GPU alias",
                    "memory_total": None,
                    "memory_used": None,
                    "utilization": None,
                },
            ],
            "duplicate",
        ),
    ],
)
def test_list_gpus_rejects_invalid_visible_ordinals(
    monkeypatch, records, message_fragment
):
    monkeypatch.setattr(server_module, "get_gpu_info", lambda: records)
    server = make_server()

    with pytest.raises(RuntimeError, match=message_fragment):
        server.list_gpus()


@pytest.mark.parametrize(
    ("record", "message_fragment"),
    [
        (
            {
                "id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": None,
            },
            "visible_id",
        ),
        (
            {
                "id": 2,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": None,
            },
            "must match",
        ),
        (
            {
                "id": -1,
                "visible_id": -1,
                "platform": "CUDA",
                "name": "GPU hidden",
                "memory_total": None,
                "memory_used": None,
                "utilization": None,
            },
            "non-negative",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": float("nan"),
            },
            "finite number",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": -1,
                "memory_used": 0,
                "utilization": None,
            },
            "'memory_total' must be a non-negative integer or null",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": 1024,
                "memory_used": -1,
                "utilization": None,
            },
            "'memory_used' must be a non-negative integer or null",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": 1024,
                "memory_used": 2048,
                "utilization": None,
            },
            "'memory_used' must not exceed 'memory_total'",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": float("inf"),
            },
            "finite number",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": -1,
            },
            "between 0 and 100",
        ),
        (
            {
                "id": 0,
                "visible_id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": 101,
            },
            "between 0 and 100",
        ),
    ],
)
def test_jsonrpc_list_gpus_rejects_malformed_gpu_record(
    monkeypatch, record, message_fragment
):
    monkeypatch.setattr(
        server_module,
        "get_gpu_info",
        lambda: [record],
    )
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 21, "method": "list_gpus", "params": {}}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 21
    assert "result" not in resp
    assert resp["error"]["code"] == JSONRPC_INTERNAL_ERROR
    assert "Malformed list_gpus response" in resp["error"]["message"]
    assert message_fragment in resp["error"]["message"]


def test_jsonrpc_list_gpus_device_enumeration_unavailable_returns_public_code(
    monkeypatch,
):
    monkeypatch.setattr(
        server_module,
        "get_gpu_info",
        lambda: (_ for _ in ()).throw(
            pm.DeviceEnumerationUnavailableError("Unable to enumerate visible GPUs")
        ),
    )
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 23, "method": "list_gpus", "params": {}}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 23
    assert "result" not in resp
    assert resp["error"]["code"] == JSONRPC_STARTUP_UNAVAILABLE
    assert "Unable to enumerate visible GPUs" in resp["error"]["message"]


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


def test_mcp_initialized_notification_with_bad_version_is_invalid_request():
    server = make_server()
    req = {"jsonrpc": "1.0", "method": "notifications/initialized"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] is None
    assert resp["error"]["code"] == JSONRPC_INVALID_REQUEST
    assert resp["error"]["message"] == "JSON-RPC version must be 2.0."


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
    for tool_name in ("start_keep", "stop_keep", "status"):
        job_id_schema = tools[tool_name]["inputSchema"]["properties"]["job_id"]
        assert job_id_schema["type"] == ["string", "null"]
        assert job_id_schema["minLength"] == 1
        assert job_id_schema["pattern"] == JOB_ID_PATTERN_TEXT


def test_jsonrpc_rejects_explicit_invalid_request_version():
    server = make_server()
    req = {"jsonrpc": "1.0", "id": 12, "method": "tools/list"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 12
    assert resp["error"]["code"] == JSONRPC_INVALID_REQUEST


def test_jsonrpc_accepts_explicit_valid_request_version():
    server = make_server()
    req = {"jsonrpc": "2.0", "id": 13, "method": "tools/list"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 13
    assert "tools" in resp["result"]


def test_jsonrpc_omitted_version_legacy_direct_call_still_works():
    server = make_server()
    req = {"id": 14, "method": "tools/list"}

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 14
    assert "tools" in resp["result"]


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


def test_mcp_tools_call_startup_unavailable_returns_tool_error():
    def failing_factory(**kwargs):
        raise SessionStartupUnavailable("No usable GPUs are available")

    server = KeepGPUServer(controller_factory=cast(Any, failing_factory))
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "start_keep",
            "arguments": {"job_id": "tool-no-gpus", "gpu_ids": [0]},
        },
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 4
    assert "result" in resp
    result = resp["result"]
    assert result["isError"] is True
    assert "No usable GPUs are available" in result["content"][0]["text"]
    assert server.status()["active_jobs"] == []


def test_mcp_tools_call_unavailable_mps_returns_tool_error(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    server = KeepGPUServer()
    req = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "start_keep",
            "arguments": {"job_id": "tool-mps-unavailable", "gpu_ids": [0]},
        },
    }

    try:
        resp = _handle_request(server, req)

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 4
        assert "result" in resp
        result = resp["result"]
        assert result["isError"] is True
        assert "PyTorch MPS backend is not available" in result["content"][0]["text"]
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_mcp_tools_call_out_of_range_gpu_ids_returns_tool_error(monkeypatch):
    import torch

    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    server = KeepGPUServer()
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "start_keep",
            "arguments": {"job_id": "tool-bad-visible-gpu", "gpu_ids": [99]},
        },
    }

    try:
        resp = _handle_request(server, req)

        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 5
        assert "result" in resp
        result = resp["result"]
        assert result["isError"] is True
        assert (
            "gpu_ids must be visible device ordinals less than 1"
            in result["content"][0]["text"]
        )
        assert server.status()["active_jobs"] == []
    finally:
        server.shutdown()


def test_mcp_tools_call_unexpected_failure_returns_jsonrpc_internal_error():
    def failing_factory(**kwargs):
        raise RuntimeError("controller exploded")

    server = KeepGPUServer(controller_factory=cast(Any, failing_factory))
    req = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "start_keep",
            "arguments": {"job_id": "tool-internal-error", "gpu_ids": [0]},
        },
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 5
    assert "result" not in resp
    assert resp["error"]["code"] == JSONRPC_INTERNAL_ERROR
    assert "controller exploded" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


def test_mcp_tools_call_list_gpus_rejects_malformed_gpu_record(monkeypatch):
    monkeypatch.setattr(
        server_module,
        "get_gpu_info",
        lambda: [
            {
                "id": 0,
                "platform": "CUDA",
                "name": "GPU 0",
                "memory_total": None,
                "memory_used": None,
                "utilization": None,
            }
        ],
    )
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 22,
        "method": "tools/call",
        "params": {"name": "list_gpus", "arguments": {}},
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 22
    assert "result" not in resp
    assert resp["error"]["code"] == JSONRPC_INTERNAL_ERROR
    assert "Malformed list_gpus response" in resp["error"]["message"]
    assert "visible_id" in resp["error"]["message"]


def test_mcp_tools_call_list_gpus_device_enumeration_unavailable_returns_tool_error(
    monkeypatch,
):
    monkeypatch.setattr(
        server_module,
        "get_gpu_info",
        lambda: (_ for _ in ()).throw(
            pm.DeviceEnumerationUnavailableError("Unable to enumerate visible GPUs")
        ),
    )
    server = make_server()
    req = {
        "jsonrpc": "2.0",
        "id": 24,
        "method": "tools/call",
        "params": {"name": "list_gpus", "arguments": {}},
    }

    resp = _handle_request(server, req)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 24
    assert "result" in resp
    result = resp["result"]
    assert result["isError"] is True
    assert "Unable to enumerate visible GPUs" in result["content"][0]["text"]


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


@pytest.mark.parametrize(
    "invalid_id",
    [True, False, None, 1.5, {"unexpected": "object"}, ["array-id"]],
)
def test_mcp_invalid_request_id_types_return_null_id(invalid_id):
    server = make_server()
    req = {"jsonrpc": "2.0", "id": invalid_id, "method": "tools/list"}

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


def test_mcp_stdio_rejects_nonstandard_json_constants_as_parse_errors():
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    )
    line = '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":NaN}\n'

    completed = subprocess.run(
        [sys.executable, "-m", "keep_gpu.mcp.server"],
        input=line,
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


def test_mcp_stdio_bad_version_notification_is_invalid_request_error():
    request = {"jsonrpc": "1.0", "method": "notifications/initialized"}
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
    assert response["id"] is None
    assert response["error"]["code"] == JSONRPC_INVALID_REQUEST
    assert response["error"]["message"] == "JSON-RPC version must be 2.0."


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


def test_stop_all_does_not_stop_session_started_after_snapshot_even_if_it_completes(
    monkeypatch,
):
    first_keep_entered = threading.Event()
    first_keep_release = threading.Event()
    stop_waiting_for_startup = threading.Event()
    second_controller = {}
    start_results = {}
    stop_result = {}

    class SlowFirstController(DummyController):
        def keep(self):
            self.kept = True
            first_keep_entered.set()
            first_keep_release.wait()

    class FastSecondController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            second_controller["value"] = self

    def factory(**kwargs):
        if kwargs.get("gpu_ids") == [1]:
            return FastSecondController(**kwargs)
        return SlowFirstController(**kwargs)

    server = KeepGPUServer(controller_factory=cast(Any, factory))
    original_wait = server._sessions_cond.wait

    def wait_for_startup(timeout=None):
        stop_waiting_for_startup.set()
        return original_wait(timeout)

    monkeypatch.setattr(server._sessions_cond, "wait", wait_for_startup)

    first_start_thread = threading.Thread(
        target=lambda: start_results.update(
            first=server.start_keep(job_id="first-job", gpu_ids=[0])
        )
    )
    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    try:
        first_start_thread.start()
        assert first_keep_entered.wait(timeout=1.0)

        stop_thread.start()
        assert stop_waiting_for_startup.wait(timeout=1.0)

        start_results["second"] = server.start_keep(job_id="second-job", gpu_ids=[1])
        assert server.status("second-job")["active"] is True

        first_keep_release.set()
        first_start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

        assert start_results["first"] == {"job_id": "first-job"}
        assert start_results["second"] == {"job_id": "second-job"}
        assert stop_result["value"]["stopped"] == ["first-job"]
        assert second_controller["value"].released is False
        assert server.status("first-job")["active"] is False
        assert server.status("second-job")["active"] is True
    finally:
        first_keep_release.set()
        first_start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)
        server.stop_keep("second-job")


def test_stop_all_does_not_stop_reused_job_id_started_after_snapshot(monkeypatch):
    starting_keep_entered = threading.Event()
    starting_keep_release = threading.Event()
    stop_waiting_for_startup = threading.Event()
    controllers_by_gpu = {}
    start_results = {}
    stop_result = {}

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            gpu_key = tuple(kwargs.get("gpu_ids") or [])
            controllers_by_gpu.setdefault(gpu_key, []).append(self)

    class BlockingStartController(TrackingController):
        def keep(self):
            self.kept = True
            starting_keep_entered.set()
            starting_keep_release.wait()

    def factory(**kwargs):
        if kwargs.get("gpu_ids") == [2]:
            return BlockingStartController(**kwargs)
        return TrackingController(**kwargs)

    server = KeepGPUServer(controller_factory=cast(Any, factory))
    original_wait = server._sessions_cond.wait

    def wait_for_startup(timeout=None):
        stop_waiting_for_startup.set()
        return original_wait(timeout)

    monkeypatch.setattr(server._sessions_cond, "wait", wait_for_startup)

    start_results["old"] = server.start_keep(job_id="reused-job", gpu_ids=[0])
    old_controller = controllers_by_gpu[(0,)][0]

    starting_thread = threading.Thread(
        target=lambda: start_results.update(
            starting=server.start_keep(job_id="starting-job", gpu_ids=[2])
        )
    )
    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    try:
        starting_thread.start()
        assert starting_keep_entered.wait(timeout=1.0)

        stop_thread.start()
        assert stop_waiting_for_startup.wait(timeout=1.0)

        targeted_stop = server.stop_keep("reused-job")
        assert targeted_stop["stopped"] == ["reused-job"]
        assert old_controller.released is True

        start_results["new"] = server.start_keep(job_id="reused-job", gpu_ids=[1])
        new_controller = controllers_by_gpu[(1,)][0]
        assert server.status("reused-job")["active"] is True

        starting_keep_release.set()
        starting_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

        assert start_results["old"] == {"job_id": "reused-job"}
        assert start_results["starting"] == {"job_id": "starting-job"}
        assert start_results["new"] == {"job_id": "reused-job"}
        assert stop_result["value"]["stopped"] == ["starting-job"]
        assert new_controller.released is False
        assert server.status("starting-job")["active"] is False
        assert server.status("reused-job")["active"] is True
    finally:
        starting_keep_release.set()
        starting_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)
        server.stop_keep("reused-job")


def test_stop_keep_times_out_waiting_for_stuck_starting_session(monkeypatch):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    start_result = {}
    stop_result = {}

    class BlockingStartController(DummyController):
        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    start_thread = threading.Thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep(job_id="starting-job"))
    )
    stop_thread.start()

    try:
        stop_thread.join(timeout=0.3)
        assert not stop_thread.is_alive()
        assert stop_result["value"]["timed_out"] == ["starting-job"]
        assert "Timed out" in stop_result["value"]["message"]
        status = server.status("starting-job")
        assert status["state"] == "stopping"
        assert status["last_error"] == server._timeout_error_message()
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

    assert start_result["value"] == {"job_id": "starting-job"}


def test_stop_all_times_out_waiting_for_stuck_starting_session(monkeypatch):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    start_result = {}
    stop_result = {}

    class BlockingStartController(DummyController):
        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    start_thread = threading.Thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    stop_thread.start()

    try:
        stop_thread.join(timeout=0.3)
        assert not stop_thread.is_alive()
        assert stop_result["value"]["timed_out"] == ["starting-job"]
        assert "Timed out" in stop_result["value"]["message"]
        status = server.status("starting-job")
        assert status["state"] == "stopping"
        assert status["last_error"] == server._timeout_error_message()
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)

    assert start_result["value"] == {"job_id": "starting-job"}


def test_timed_out_stop_of_starting_session_reports_stopping_before_startup_finishes(
    monkeypatch,
):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    start_result = {}

    class BlockingStartController(DummyController):
        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    start_thread = threading.Thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    try:
        stop_result = server.stop_keep(job_id="starting-job")

        assert stop_result["timed_out"] == ["starting-job"]
        expected_error = server._timeout_error_message()
        status = server.status("starting-job")
        assert status["active"] is True
        assert status["state"] == "stopping"
        assert status["last_error"] == expected_error
        jobs = {job["job_id"]: job for job in server.status()["active_jobs"]}
        assert jobs["starting-job"]["state"] == "stopping"
        assert jobs["starting-job"]["last_error"] == expected_error
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)

    assert start_result["value"] == {"job_id": "starting-job"}


def test_pending_stop_session_is_not_reported_with_active_state_before_background_release(
    monkeypatch,
):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    controllers = []
    deferred_threads = []
    start_result = {}

    class BlockingStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    real_thread = threading.Thread

    class DeferredThread:
        def __init__(self, target, args=(), kwargs=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            deferred_threads.append(self)

        def run(self):
            self._target(*self._args, **self._kwargs)

    def thread_factory(*args, **kwargs):
        target_kwargs = kwargs.get("kwargs") or {}
        if (
            target_kwargs.get("job_id") == "starting-job"
            and target_kwargs.get("expected_session") is not None
        ):
            return DeferredThread(
                kwargs.get("target"), kwargs.get("args", ()), target_kwargs
            )
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(server_module.threading, "Thread", thread_factory)

    start_thread = real_thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    try:
        stop_result = server.stop_keep(job_id="starting-job")
        assert stop_result["timed_out"] == ["starting-job"]

        keep_release.set()
        start_thread.join(timeout=1.0)
        assert start_result["value"] == {"job_id": "starting-job"}
        assert len(deferred_threads) == 1

        expected_error = server._timeout_error_message()
        status = server.status("starting-job")
        # "active" means the retained session is visible; state carries lifecycle truth.
        assert status["active"] is True
        assert status["state"] == "stopping"
        assert status["last_error"] == expected_error

        duplicate_stop = server.stop_keep("starting-job")
        assert duplicate_stop["timed_out"] == ["starting-job"]
        assert controllers[0].released is False
        assert server.status("starting-job")["last_error"] == expected_error

        deferred_threads[0].run()
        assert controllers[0].released is True
        assert server.status("starting-job")["active"] is False
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)


def test_timed_out_stop_of_starting_session_releases_after_startup_completes(
    monkeypatch,
):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    controllers = []
    start_result = {}
    stop_result = {}

    class BlockingStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)
    info_messages = []
    original_info = server_module.logger.info

    def record_info(message, *args, **kwargs):
        info_messages.append(message % args if args else message)
        original_info(message, *args, **kwargs)

    monkeypatch.setattr(server_module.logger, "info", record_info)

    start_thread = threading.Thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep(job_id="starting-job"))
    )
    stop_thread.start()

    try:
        stop_thread.join(timeout=0.3)
        assert not stop_thread.is_alive()
        assert stop_result["value"]["timed_out"] == ["starting-job"]
        assert controllers[0].released is False

        keep_release.set()
        start_thread.join(timeout=1.0)
        assert start_result["value"] == {"job_id": "starting-job"}
        assert _wait_until(lambda: server.status("starting-job")["active"] is False)
        assert controllers[0].released is True
        assert "Stopped keep session starting-job" in info_messages
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)


def test_pending_stop_does_not_stop_reused_job_id_after_original_removed(
    monkeypatch,
):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    controllers_by_gpu = {}
    deferred_threads = []
    start_result = {}

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            gpu_key = tuple(kwargs.get("gpu_ids") or [])
            controllers_by_gpu.setdefault(gpu_key, []).append(self)

    class BlockingStartController(TrackingController):
        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    def factory(**kwargs):
        if kwargs.get("gpu_ids") == [0]:
            return BlockingStartController(**kwargs)
        return TrackingController(**kwargs)

    server = KeepGPUServer(controller_factory=cast(Any, factory))
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    real_thread = threading.Thread

    class DeferredThread:
        def __init__(self, target, kwargs):
            self._target = target
            self._kwargs = kwargs

        def start(self):
            deferred_threads.append(self)

        def run(self):
            self._target(**self._kwargs)

    def thread_factory(*args, **kwargs):
        target = kwargs.get("target")
        target_kwargs = kwargs.get("kwargs") or {}
        if (
            target_kwargs.get("job_id") == "race-job"
            and target_kwargs.get("expected_session") is not None
        ):
            return DeferredThread(target, target_kwargs)
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(server_module.threading, "Thread", thread_factory)

    start_thread = real_thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="race-job", gpu_ids=[0])
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    timed_out_stop = server.stop_keep(job_id="race-job")
    assert timed_out_stop["timed_out"] == ["race-job"]

    keep_release.set()
    start_thread.join(timeout=1.0)
    assert start_result["value"] == {"job_id": "race-job"}
    assert len(deferred_threads) == 1

    old_controller = controllers_by_gpu[(0,)][0]
    old_session = server._sessions["race-job"]
    stop_original = server._stop_current_session(
        "race-job", expected_session=old_session, pending_stop_cleanup=True
    )
    assert stop_original["stopped"] == ["race-job"]
    assert old_controller.released is True

    replacement_start = server.start_keep(job_id="race-job", gpu_ids=[1])
    replacement_controller = controllers_by_gpu[(1,)][0]
    assert replacement_start == {"job_id": "race-job"}
    assert server.status("race-job")["active"] is True

    deferred_threads[0].run()

    assert replacement_controller.released is False
    assert server.status("race-job")["active"] is True
    server.stop_keep("race-job")


def test_stop_keep_does_not_stop_reused_job_id_after_wait_window(monkeypatch):
    controllers_by_gpu = {}
    start_results = {}
    intercept_wait = True

    class TrackingController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            gpu_key = tuple(kwargs.get("gpu_ids") or [])
            controllers_by_gpu.setdefault(gpu_key, []).append(self)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: TrackingController(**kwargs))
    )

    start_results["old"] = server.start_keep(job_id="race-job", gpu_ids=[0])
    old_controller = controllers_by_gpu[(0,)][0]
    original_wait = server._wait_for_starting_jobs_or_mark_pending

    def replace_job_id_during_wait(starting_snapshot):
        nonlocal intercept_wait
        if not intercept_wait:
            return original_wait(starting_snapshot)
        intercept_wait = False
        result = original_wait(starting_snapshot)
        stop_original = server.stop_keep("race-job")
        assert stop_original["stopped"] == ["race-job"]
        start_results["new"] = server.start_keep(job_id="race-job", gpu_ids=[1])
        return result

    monkeypatch.setattr(
        server, "_wait_for_starting_jobs_or_mark_pending", replace_job_id_during_wait
    )

    stop_result = server.stop_keep("race-job")

    replacement_controller = controllers_by_gpu[(1,)][0]
    assert start_results["old"] == {"job_id": "race-job"}
    assert start_results["new"] == {"job_id": "race-job"}
    assert old_controller.released is True
    assert stop_result["message"] == "job_id not found"
    assert replacement_controller.released is False
    assert server.status("race-job")["active"] is True
    server.stop_keep("race-job")


def test_timed_out_stop_all_of_starting_session_releases_after_startup_completes(
    monkeypatch,
):
    keep_entered = threading.Event()
    keep_release = threading.Event()
    controllers = []
    start_result = {}
    stop_result = {}

    class BlockingStartController(DummyController):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            controllers.append(self)

        def keep(self):
            self.kept = True
            keep_entered.set()
            keep_release.wait(timeout=1.0)

    server = KeepGPUServer(
        controller_factory=cast(Any, lambda **kwargs: BlockingStartController(**kwargs))
    )
    monkeypatch.setattr(server, "_startup_stop_wait_timeout_s", 0.02, raising=False)

    start_thread = threading.Thread(
        target=lambda: start_result.update(
            value=server.start_keep(job_id="starting-job")
        )
    )
    start_thread.start()
    assert keep_entered.wait(timeout=1.0)

    stop_thread = threading.Thread(
        target=lambda: stop_result.update(value=server.stop_keep())
    )
    stop_thread.start()

    try:
        stop_thread.join(timeout=0.3)
        assert not stop_thread.is_alive()
        assert stop_result["value"]["timed_out"] == ["starting-job"]
        assert controllers[0].released is False

        keep_release.set()
        start_thread.join(timeout=1.0)
        assert start_result["value"] == {"job_id": "starting-job"}
        assert _wait_until(lambda: server.status("starting-job")["active"] is False)
        assert controllers[0].released is True
    finally:
        keep_release.set()
        start_thread.join(timeout=1.0)
        stop_thread.join(timeout=1.0)


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
