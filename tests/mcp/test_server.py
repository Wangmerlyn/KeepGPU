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


def test_start_status_stop_cycle():
    server = KeepGPUServer(controller_factory=dummy_factory)
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


def test_stop_all():
    server = KeepGPUServer(controller_factory=dummy_factory)
    job_a = server.start_keep()["job_id"]
    job_b = server.start_keep()["job_id"]

    stopped = server.stop_keep()
    assert set(stopped["stopped"]) == {job_a, job_b}
    assert server.status(job_a)["active"] is False
    assert server.status(job_b)["active"] is False


def test_list_gpus():
    server = KeepGPUServer(controller_factory=dummy_factory)
    info = server.list_gpus()
    assert "gpus" in info


def test_end_to_end_jsonrpc():
    server = KeepGPUServer(controller_factory=dummy_factory)
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
    server = KeepGPUServer(controller_factory=dummy_factory)
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
