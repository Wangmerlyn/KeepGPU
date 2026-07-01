import threading

import pytest

from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


def test_rocm_keep_raises_when_worker_startup_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
    )

    def fail_set_device(_rank):
        raise RuntimeError("rocm startup failed")

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", fail_set_device)

    with pytest.raises(RuntimeError, match="rocm startup failed"):
        ctrl.keep()

    assert not (ctrl._thread and ctrl._thread.is_alive())


def test_rocm_keep_raises_when_startup_allocation_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
        iterations=1,
    )
    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("rocm startup allocation failed")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    with pytest.raises(RuntimeError, match="rocm startup allocation failed"):
        ctrl.keep()

    assert ctrl._thread is None
    assert ctrl._stop_evt is None


def test_rocm_keep_raises_when_startup_oom_exhausts_retry_budget(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(rocm_module.torch.cuda, "empty_cache", lambda: None)

    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
        iterations=1,
        max_allocation_retries=1,
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("ROCm out of memory")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    with pytest.raises(
        RuntimeError, match="failed to allocate tensor after 1 attempts"
    ):
        ctrl.keep()

    assert ctrl._thread is None
    assert ctrl._stop_evt is None


def test_rocm_keep_returns_when_startup_defers_for_unknown_utilization(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(rocm_module.torch.cuda, "empty_cache", lambda: None)

    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=25,
        iterations=1,
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: None)

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("allocation should be deferred while telemetry is unknown")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl.keep()

    assert ctrl._thread is not None
    assert ctrl._thread.is_alive()
    assert ctrl.allocation_status() is None
    ctrl.release()
    assert ctrl._thread is None


def test_rocm_keep_returns_for_recoverable_startup_oom_retry(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(rocm_module.torch.cuda, "empty_cache", lambda: None)

    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
        iterations=1,
        max_allocation_retries=None,
    )
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("ROCm out of memory")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl.keep()

    assert ctrl._thread is not None
    assert ctrl._thread.is_alive()
    assert ctrl.allocation_status() is None
    ctrl.release()
    assert ctrl._thread is None


def test_rocm_keep_shuts_down_smi_when_worker_startup_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)
    calls = []

    class DummyRocmSmi:
        def rsmi_init(self):
            calls.append("init")

        def rsmi_shut_down(self):
            calls.append("shutdown")

    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
    )
    ctrl._rocm_smi = DummyRocmSmi()

    def fail_set_device(_rank):
        raise RuntimeError("rocm startup failed")

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", fail_set_device)

    with pytest.raises(RuntimeError, match="rocm startup failed"):
        ctrl.keep()

    assert calls == ["init", "shutdown"]


def test_rocm_keep_rejects_retry_while_startup_thread_is_stopping(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)

    class AliveThread:
        def is_alive(self):
            return True

    ctrl = RocmGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
    )
    ctrl._thread = AliveThread()
    ctrl._stop_evt = threading.Event()
    ctrl._stop_evt.set()

    with pytest.raises(RuntimeError, match="startup did not complete"):
        ctrl.keep()


@pytest.mark.parametrize("iterations", [0, -1])
def test_rocm_controller_rejects_non_positive_iterations(iterations):
    with pytest.raises(ValueError, match="iterations must be positive"):
        RocmGPUController(rank=0, vram_to_keep=4, iterations=iterations)


@pytest.mark.parametrize(
    ("max_allocation_retries", "error_type", "message"),
    [
        ("1", TypeError, "max_allocation_retries must be an integer"),
        (True, TypeError, "max_allocation_retries must be an integer"),
        (0, ValueError, "max_allocation_retries must be positive"),
        (-1, ValueError, "max_allocation_retries must be positive"),
    ],
)
def test_rocm_controller_rejects_invalid_max_allocation_retries(
    monkeypatch, max_allocation_retries, error_type, message
):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(error_type, match=message):
        RocmGPUController(
            rank=0,
            vram_to_keep=4,
            max_allocation_retries=max_allocation_retries,
        )


def test_rocm_controller_accepts_positive_max_allocation_retries(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module.torch.cuda, "device_count", lambda: 1)

    ctrl = RocmGPUController(rank=0, vram_to_keep=4, max_allocation_retries=1)

    assert ctrl.max_allocation_retries == 1


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


class _StopWaitForbidden:
    def is_set(self):
        return False

    def wait(self, _timeout):
        raise AssertionError("fatal worker failures should stop immediately")


def test_rocm_unknown_utilization_backs_off_when_threshold_enabled():
    assert RocmGPUController._should_run_batch(None, 10) is False


@pytest.mark.parametrize("utilization", [-1, 101, False])
def test_rocm_invalid_utilization_backs_off_when_threshold_enabled(utilization):
    assert RocmGPUController._should_run_batch(utilization, 10) is False


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


def test_rocm_records_unexpected_post_start_worker_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(rocm_module.torch, "rand", lambda *args, **kwargs: object())
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_batch(_tensor):
        raise ValueError("fatal compute exploded")

    monkeypatch.setattr(ctrl, "_run_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected ROCm keep worker failure: fatal compute exploded"
    )


def test_rocm_records_post_start_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(rocm_module.torch, "rand", lambda *args, **kwargs: object())
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_batch(_tensor):
        raise RuntimeError("rocm backend exploded")

    monkeypatch.setattr(ctrl, "_run_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected ROCm keep worker failure: rocm backend exploded"
    )


def test_rocm_records_post_start_allocation_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("rocm allocation exploded")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected ROCm keep worker failure: rocm allocation exploded"
    )


def test_rocm_sets_startup_event_when_recording_failure_without_error_list(
    monkeypatch,
):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()
    startup_evt = threading.Event()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("rocm allocation exploded")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop(startup_evt=startup_evt, startup_errors=None)

    assert startup_evt.is_set()
    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected ROCm keep worker failure: rocm allocation exploded"
    )


def test_rocm_sets_startup_event_when_retry_exhausts_without_error_list(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()
    startup_evt = threading.Event()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("ROCm out of memory")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop(startup_evt=startup_evt, startup_errors=None)

    assert startup_evt.is_set()
    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert str(error) == "rank 0: failed to allocate tensor after 1 attempts"


def test_rocm_records_unexpected_post_start_allocation_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.max_allocation_retries = None
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(rocm_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_query_utilization", lambda: 0)

    def fail_allocation(*args, **kwargs):
        raise ValueError("allocator corrupted")

    monkeypatch.setattr(rocm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error) == "rank 0: unexpected ROCm keep worker failure: allocator corrupted"
    )
