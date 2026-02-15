import pytest
import time
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from tests.polling import wait_until


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
