import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from tests.polling import wait_until


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_context_manager():
    """Validate VRAM target consumption during keep and recovery after release."""
    ctrl = CudaGPUController(
        rank=torch.cuda.device_count() - 1,
        interval=0.05,
        vram_to_keep="64MB",
        relu_iterations=64,
    )

    torch.cuda.set_device(ctrl.rank)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    target_bytes = int(ctrl.vram_to_keep) * 4  # float32 bytes
    free_bytes, _ = torch.cuda.mem_get_info(ctrl.rank)
    if free_bytes < int(target_bytes * 1.2):
        pytest.skip(
            f"Insufficient free VRAM for assertion test: need ~{target_bytes}, have {free_bytes}"
        )

    alloc_tolerance = 8 * 1024 * 1024
    reserve_tolerance = 16 * 1024 * 1024
    before_reserved = torch.cuda.memory_reserved(ctrl.rank)
    before_allocated = torch.cuda.memory_allocated(ctrl.rank)

    with ctrl:
        assert ctrl._thread and ctrl._thread.is_alive()
        peak_alloc_delta = 0
        peak_reserved_delta = 0

        def _target_reached() -> bool:
            nonlocal peak_alloc_delta, peak_reserved_delta
            alloc_delta = max(
                0, torch.cuda.memory_allocated(ctrl.rank) - before_allocated
            )
            reserved_delta = max(
                0, torch.cuda.memory_reserved(ctrl.rank) - before_reserved
            )
            peak_alloc_delta = max(peak_alloc_delta, alloc_delta)
            peak_reserved_delta = max(peak_reserved_delta, reserved_delta)
            # allocated should track payload; reserved may be larger due allocator blocks
            return (
                alloc_delta >= int(target_bytes * 0.95)
                and reserved_delta >= alloc_delta
            )

        reached_target = wait_until(_target_reached, timeout_s=3.0)

        assert reached_target, (
            f"VRAM target not reached. target={target_bytes}, "
            f"peak_alloc_delta={peak_alloc_delta}, peak_reserved_delta={peak_reserved_delta}"
        )

    alloc_delta_after = -1
    reserved_delta_after = -1

    def _released() -> bool:
        nonlocal alloc_delta_after, reserved_delta_after
        alloc_after = torch.cuda.memory_allocated(ctrl.rank)
        reserved_after = torch.cuda.memory_reserved(ctrl.rank)
        alloc_delta_after = max(0, alloc_after - before_allocated)
        reserved_delta_after = max(0, reserved_after - before_reserved)
        return (
            alloc_delta_after <= alloc_tolerance
            and reserved_delta_after <= reserve_tolerance
            and not (ctrl._thread and ctrl._thread.is_alive())
        )

    released = wait_until(_released, timeout_s=3.0)

    assert released, (
        "VRAM did not return near baseline after release. "
        f"alloc_delta_after={alloc_delta_after}, reserved_delta_after={reserved_delta_after}"
    )
