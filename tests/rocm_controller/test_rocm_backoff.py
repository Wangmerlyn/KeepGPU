import pytest

from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


@pytest.mark.parametrize("iterations", [0, -1])
def test_rocm_controller_rejects_non_positive_iterations(iterations):
    with pytest.raises(ValueError, match="iterations must be positive"):
        RocmGPUController(rank=0, vram_to_keep=4, iterations=iterations)


class _StopAfterOneWait:
    def __init__(self):
        self.stopped = False
        self.wait_calls = 0

    def is_set(self):
        return self.stopped

    def wait(self, _timeout):
        self.wait_calls += 1
        self.stopped = True
        return True


class _StopAfterWaits:
    def __init__(self, stop_after):
        self.stop_after = stop_after
        self.stopped = False
        self.wait_calls = 0

    def is_set(self):
        return self.stopped

    def wait(self, _timeout):
        self.wait_calls += 1
        if self.wait_calls >= self.stop_after:
            self.stopped = True
        return self.stopped


def test_rocm_unknown_utilization_backs_off_when_threshold_enabled():
    assert RocmGPUController._should_run_batch(None, 10) is False


def test_rocm_unknown_utilization_runs_when_backoff_disabled():
    assert RocmGPUController._should_run_batch(None, -1) is True


def test_rocm_busy_utilization_defers_initial_allocation(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterOneWait()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        rocm_module.torch,
        "rand",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("allocation should wait for idle telemetry")
        ),
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 100)

    ctrl._keep_loop()

    assert ctrl._stop_evt.wait_calls == 1


def test_rocm_busy_then_idle_utilization_allows_initial_allocation(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterWaits(stop_after=2)

    allocations = []
    batches = []
    utilization = iter([100, 0, 0])

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        rocm_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: next(utilization))
    monkeypatch.setattr(ctrl, "_run_batch", lambda tensor: batches.append(tensor))

    ctrl._keep_loop()

    assert len(allocations) == 1
    assert len(batches) == 1
    assert ctrl._stop_evt.wait_calls == 2


def test_rocm_busy_deferrals_do_not_consume_allocation_retries(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.iterations = 1
    ctrl.max_allocation_retries = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopAfterWaits(stop_after=3)

    allocations = []
    batches = []
    utilization = iter([100, 100, 0, 0])

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        rocm_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: next(utilization))
    monkeypatch.setattr(ctrl, "_run_batch", lambda tensor: batches.append(tensor))

    ctrl._keep_loop()

    assert ctrl._failure_exc is None
    assert len(allocations) == 1
    assert len(batches) == 1
    assert ctrl._stop_evt.wait_calls == 3


def test_rocm_negative_busy_threshold_allocates_even_when_busy(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterWaits(stop_after=1)

    allocations = []

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        rocm_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 100)
    monkeypatch.setattr(ctrl, "_run_batch", lambda _tensor: None)

    ctrl._keep_loop()

    assert len(allocations) == 1
    assert ctrl._stop_evt.wait_calls == 1
