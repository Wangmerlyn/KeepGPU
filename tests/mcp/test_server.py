import threading
import time
from typing import Any, cast

from keep_gpu.mcp.server import KeepGPUServer, _handle_request


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


def test_start_rejects_negative_gpu_id():
    server = make_server()

    try:
        server.start_keep(gpu_ids=[0, -1])
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "gpu_ids must contain non-negative integers" in str(exc)

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
    assert "interval must be positive" in resp["error"]["message"]
    assert server.status()["active_jobs"] == []


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

    outcomes = iter([True, False])
    monkeypatch.setattr(
        server,
        "_release_with_timeout",
        lambda controller, **_: next(outcomes),
    )

    result = server.stop_keep()
    assert result["stopped"] == [job_a]
    assert result["timed_out"] == [job_b]
    assert server.status(job_a)["active"] is False
    status_b = server.status(job_b)
    assert status_b["active"] is True
    assert status_b["state"] == "stopping"


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
