import time

import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


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


def test_negative_busy_threshold_disables_backoff_without_gpu():
    assert CudaGPUController._should_run_batch(0, -1) is True
    assert CudaGPUController._should_run_batch(100, -1) is True
    assert CudaGPUController._should_run_batch(None, -1) is True


def test_non_negative_busy_threshold_backs_off_above_limit_without_gpu():
    assert CudaGPUController._should_run_batch(10, 10) is True
    assert CudaGPUController._should_run_batch(11, 10) is False
    assert CudaGPUController._should_run_batch(None, 10) is False


@pytest.mark.parametrize("utilization", [-1, 101, False])
def test_non_negative_busy_threshold_treats_invalid_utilization_as_unknown(
    utilization,
):
    assert CudaGPUController._should_run_batch(utilization, 10) is False


def test_cuda_monitor_preserves_unknown_utilization(monkeypatch):
    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.cuda_gpu_controller.get_gpu_utilization",
        lambda rank: None,
    )

    assert CudaGPUController._monitor_utilization(0) is None


def test_cuda_controller_rejects_busy_threshold_below_minus_one_without_gpu():
    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        CudaGPUController(rank=0, vram_to_keep="4MB", busy_threshold=-2)


def test_cuda_busy_utilization_defers_initial_allocation(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterOneWait()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        cuda_module.torch,
        "rand",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("allocation should wait for idle telemetry")
        ),
    )
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 100)

    ctrl._keep_loop()

    assert ctrl._stop_evt.wait_calls == 1


def test_cuda_busy_then_idle_utilization_allows_initial_allocation(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = 10
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterWaits(stop_after=2)

    allocations = []
    batches = []
    utilization = iter([100, 0, 0])

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        cuda_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: next(utilization))
    monkeypatch.setattr(ctrl, "_run_relu_batch", lambda matrix: batches.append(matrix))

    ctrl._keep_loop()

    assert len(allocations) == 1
    assert len(batches) == 1
    assert ctrl._stop_evt.wait_calls == 2


def test_cuda_negative_busy_threshold_allocates_even_when_busy(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._stop_evt = _StopAfterWaits(stop_after=1)

    allocations = []

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        cuda_module.torch,
        "rand",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 100)
    monkeypatch.setattr(ctrl, "_run_relu_batch", lambda _matrix: None)

    ctrl._keep_loop()

    assert len(allocations) == 1
    assert ctrl._stop_evt.wait_calls == 1


def test_cuda_records_unexpected_post_start_worker_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(cuda_module.torch, "rand", lambda *args, **kwargs: object())
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_batch(_matrix):
        raise ValueError("fatal compute exploded")

    monkeypatch.setattr(ctrl, "_run_relu_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected CUDA keep worker failure: fatal compute exploded"
    )


def test_cuda_records_post_start_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(cuda_module.torch, "rand", lambda *args, **kwargs: object())
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_batch(_matrix):
        raise RuntimeError("device-side assert")

    monkeypatch.setattr(ctrl, "_run_relu_batch", fail_batch)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error) == "rank 0: unexpected CUDA keep worker failure: device-side assert"
    )


def test_cuda_records_unexpected_post_start_allocation_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_allocation(*args, **kwargs):
        raise ValueError("allocator corrupted")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error) == "rank 0: unexpected CUDA keep worker failure: allocator corrupted"
    )


def test_cuda_records_post_start_allocation_runtime_error_as_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("illegal memory access")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert (
        str(error)
        == "rank 0: unexpected CUDA keep worker failure: illegal memory access"
    )


def test_cuda_retries_post_start_allocation_oom_without_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopAfterOneWait()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_allocation(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    ctrl._keep_loop()

    assert ctrl.allocation_status() is None
    assert ctrl._stop_evt.wait_calls == 1


def test_cuda_retries_steady_state_oom_without_failure(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl._num_elements = 4
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopAfterOneWait()

    cache_calls = []

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        cuda_module.torch.cuda, "empty_cache", lambda: cache_calls.append("empty_cache")
    )
    monkeypatch.setattr(cuda_module.torch, "rand", lambda *args, **kwargs: object())
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_batch(_matrix):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(ctrl, "_run_relu_batch", fail_batch)

    ctrl._keep_loop()

    assert ctrl.allocation_status() is None
    assert ctrl._stop_evt.wait_calls == 1
    assert cache_calls == ["empty_cache"]


def test_cuda_records_invalid_post_start_num_elements_without_startup_errors(
    monkeypatch,
):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.device = "cuda:0"
    ctrl.interval = 0.01
    ctrl.busy_threshold = -1
    ctrl.relu_iterations = 1
    ctrl.vram_to_keep = 0
    ctrl._num_elements = 0
    ctrl._failure_exc = None
    ctrl._stop_evt = _StopWaitForbidden()

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)

    ctrl._keep_loop()

    error = ctrl.allocation_status()
    assert isinstance(error, RuntimeError)
    assert str(error) == "rank 0: invalid vram_to_keep=0"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_controller_respects_busy_threshold(monkeypatch):
    calls = {"run": 0}

    ctrl = CudaGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="4MB",
        busy_threshold=10,
        relu_iterations=8,
    )

    def fake_utilization(_rank: int) -> int:
        return 100  # always above threshold

    def fake_run(matrix):
        calls["run"] += 1

    monkeypatch.setattr(ctrl, "_monitor_utilization", fake_utilization)
    monkeypatch.setattr(ctrl, "_run_relu_batch", fake_run)

    ctrl.keep()
    time.sleep(0.2)
    ctrl.release()

    # When utilization is always high, no ReLU keep-alive batches should run.
    assert calls["run"] == 0
