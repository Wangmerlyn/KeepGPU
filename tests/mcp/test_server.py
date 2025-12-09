from keep_gpu.mcp.server import KeepGPUServer


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

