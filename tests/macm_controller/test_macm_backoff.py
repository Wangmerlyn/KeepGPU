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
