import threading
import time

import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from tests.polling import wait_until


def test_cuda_keep_raises_when_worker_startup_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)

    ctrl = CudaGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
    )

    def fail_set_device(_rank):
        raise RuntimeError("cuda startup failed")

    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", fail_set_device)

    with pytest.raises(RuntimeError, match="cuda startup failed"):
        ctrl.keep()

    assert not (ctrl._thread and ctrl._thread.is_alive())


def test_cuda_keep_raises_when_startup_allocation_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)

    ctrl = CudaGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
        relu_iterations=1,
    )
    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("cuda startup allocation failed")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    with pytest.raises(RuntimeError, match="cuda startup allocation failed"):
        ctrl.keep()

    assert ctrl._thread is None
    assert ctrl._stop_evt is None


def test_cuda_keep_clears_state_when_thread_start_fails(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(cuda_module.threading, "Thread", FailingThread)

    ctrl = CudaGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=-1,
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        ctrl.keep()

    assert ctrl._thread is None
    assert ctrl._stop_evt is None


def test_cuda_keep_returns_when_startup_defers_for_unknown_utilization(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(cuda_module.torch.cuda, "empty_cache", lambda: None)

    ctrl = CudaGPUController(
        rank=0,
        interval=0.01,
        vram_to_keep=4,
        busy_threshold=25,
        relu_iterations=1,
    )
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: None)

    def fail_allocation(*_args, **_kwargs):
        raise AssertionError("allocation should be deferred while telemetry is unknown")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    ctrl.keep()

    assert ctrl._thread is not None
    assert ctrl._thread.is_alive()
    assert ctrl.allocation_status() is None
    ctrl.release()
    assert ctrl._thread is None


def test_cuda_keep_returns_for_recoverable_startup_oom_retry(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    calls = []
    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cuda_module.torch.cuda, "set_device", lambda _rank: None)
    monkeypatch.setattr(
        cuda_module.torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )

    ctrl = CudaGPUController(
        rank=0,
        interval=60,
        vram_to_keep=4,
        busy_threshold=-1,
        relu_iterations=1,
    )
    monkeypatch.setattr(ctrl, "_monitor_utilization", lambda _rank: 0)

    def fail_allocation(*_args, **_kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cuda_module.torch, "rand", fail_allocation)

    ctrl.keep()

    try:
        assert ctrl._thread is not None
        assert ctrl._thread.is_alive()
        assert ctrl.allocation_status() is None
        assert calls == ["empty_cache"]
    finally:
        ctrl.release()
    assert ctrl._thread is None


def test_cuda_keep_rejects_retry_while_startup_thread_is_stopping(monkeypatch):
    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    class AliveThread:
        def is_alive(self):
            return True

    monkeypatch.setattr(cuda_module.torch.cuda, "device_count", lambda: 1)

    ctrl = CudaGPUController(
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_basic():
    ctrl = CudaGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="8MB",
        relu_iterations=64,
    )
    ctrl.keep()
    time.sleep(0.2)
    assert ctrl._thread and ctrl._thread.is_alive()

    ctrl.release()
    assert not (ctrl._thread and ctrl._thread.is_alive())

    ctrl.keep()
    time.sleep(0.2)
    assert ctrl._thread and ctrl._thread.is_alive()
    ctrl.release()
    assert not (ctrl._thread and ctrl._thread.is_alive())

    with ctrl:
        assert ctrl._thread and ctrl._thread.is_alive()
        time.sleep(0.2)
    assert not (ctrl._thread and ctrl._thread.is_alive())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_respects_vram_target_during_keep():
    """Ensure keep() consumes roughly requested VRAM and release() frees it."""
    ctrl = CudaGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="32MB",
        relu_iterations=32,
    )
    torch.cuda.set_device(ctrl.rank)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    target_bytes = int(ctrl.vram_to_keep) * 4
    free_bytes, _ = torch.cuda.mem_get_info(ctrl.rank)
    if free_bytes < int(target_bytes * 1.2):
        pytest.skip(
            f"Insufficient free VRAM for assertion test: need ~{target_bytes}, have {free_bytes}"
        )

    before_alloc = torch.cuda.memory_allocated(ctrl.rank)
    before_reserved = torch.cuda.memory_reserved(ctrl.rank)
    alloc_tolerance = 8 * 1024 * 1024
    reserve_tolerance = 16 * 1024 * 1024

    ctrl.keep()
    reached = wait_until(
        lambda: (
            (
                alloc_delta := max(
                    0, torch.cuda.memory_allocated(ctrl.rank) - before_alloc
                )
            )
            >= int(target_bytes * 0.95)
            and max(0, torch.cuda.memory_reserved(ctrl.rank) - before_reserved)
            >= alloc_delta
        ),
        timeout_s=3.0,
    )
    assert reached, "keep() did not reach expected VRAM allocation target in time"

    alloc_delta = max(0, torch.cuda.memory_allocated(ctrl.rank) - before_alloc)
    reserved_delta = max(0, torch.cuda.memory_reserved(ctrl.rank) - before_reserved)
    assert alloc_delta >= int(target_bytes * 0.95)
    assert reserved_delta >= alloc_delta

    ctrl.release()
    released = wait_until(
        lambda: (
            max(0, torch.cuda.memory_allocated(ctrl.rank) - before_alloc)
            <= alloc_tolerance
            and max(0, torch.cuda.memory_reserved(ctrl.rank) - before_reserved)
            <= reserve_tolerance
        ),
        timeout_s=3.0,
    )
    assert released, "VRAM did not return near baseline after release()"
