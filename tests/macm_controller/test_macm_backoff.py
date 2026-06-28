from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController


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


def test_macm_unknown_utilization_backs_off_when_threshold_enabled():
    assert MacMGPUController._should_run_batch(None, 10) is False


def test_macm_unknown_utilization_runs_when_backoff_disabled():
    assert MacMGPUController._should_run_batch(None, -1) is True


def test_macm_unavailable_utilization_defers_initial_allocation(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterOneWait()

    monkeypatch.setattr(
        macm_module.torch,
        "rand",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("allocation should wait for idle telemetry")
        ),
    )

    ctrl._keep_loop()

    assert ctrl._stop_evt.wait_calls == 1


def test_macm_negative_busy_threshold_allocates_with_unavailable_telemetry(
    monkeypatch,
):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterWaits(stop_after=1)

    allocations = []

    monkeypatch.setattr(
        macm_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_run_batch", lambda _tensor: None)

    ctrl._keep_loop()

    assert len(allocations) == 1
    assert ctrl._stop_evt.wait_calls == 1


def test_macm_records_unexpected_post_start_worker_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(macm_module.torch, "rand", lambda *args, **kwargs: object())

    def fail_batch(_tensor):
        raise ValueError("fatal compute exploded")

    monkeypatch.setattr(ctrl, "_run_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected MPS keep worker failure: fatal compute exploded"
    )


def test_macm_records_post_start_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(macm_module.torch, "rand", lambda *args, **kwargs: object())

    def fail_batch(_tensor):
        raise RuntimeError("mps backend exploded")

    monkeypatch.setattr(ctrl, "_run_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error) == "rank 0: unexpected MPS keep worker failure: mps backend exploded"
    )


def test_macm_records_unexpected_post_start_allocation_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    def fail_allocation(*args, **kwargs):
        raise ValueError("allocator corrupted")

    monkeypatch.setattr(macm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error) == "rank 0: unexpected MPS keep worker failure: allocator corrupted"
    )


def test_macm_records_post_start_allocation_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("mps allocation exploded")

    monkeypatch.setattr(macm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected MPS keep worker failure: mps allocation exploded"
    )


def test_macm_records_invalid_post_start_num_elements_as_failure():
    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl.vram_to_keep = 0
    ctrl._num_elements = 0
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert str(error) == "rank 0: invalid vram_to_keep=0"


def test_macm_retries_post_start_allocation_oom_without_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopAfterOneWait()

    cache_calls = []

    monkeypatch.setattr(
        macm_module.torch.mps,
        "empty_cache",
        lambda: cache_calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.macm_gpu_controller.gc.collect",
        lambda: cache_calls.append("gc.collect"),
    )

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(macm_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    assert ctrl.allocation_status() is None
    assert ctrl._stop_evt.wait_calls == 1
    assert cache_calls == ["empty_cache", "gc.collect"]


def test_macm_retries_steady_state_oom_without_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 0
    ctrl.device = "mps"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopAfterOneWait()

    cache_calls = []

    monkeypatch.setattr(
        macm_module.torch.mps,
        "empty_cache",
        lambda: cache_calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.macm_gpu_controller.gc.collect",
        lambda: cache_calls.append("gc.collect"),
    )
    monkeypatch.setattr(macm_module.torch, "rand", lambda *args, **kwargs: object())

    def fail_batch(_tensor):
        raise RuntimeError("MPS out of memory")

    monkeypatch.setattr(ctrl, "_run_batch", fail_batch)

    ctrl._keep_loop()

    assert ctrl.allocation_status() is None
    assert ctrl._stop_evt.wait_calls == 1
    assert cache_calls == ["empty_cache", "gc.collect"]
