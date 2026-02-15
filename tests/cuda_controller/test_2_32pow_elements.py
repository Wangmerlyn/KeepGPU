import time
import pytest
import torch
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.large_memory
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_vram_allocation():
    """Tests controller with a large VRAM allocation."""
    # Intentionally using full 2**32 float32 elements (~16 GiB) for large-tensor testing.
    # Torch may expose indexing issues around this boundary on some systems.
    vram_elements = 2**32
    required_bytes = vram_elements * 4
    free_bytes, _ = torch.cuda.mem_get_info(0)
    if free_bytes < required_bytes:
        pytest.skip(
            f"Insufficient free VRAM for large test: need {required_bytes}, have {free_bytes}"
        )

    controller = CudaGPUController(
        rank=0,
        interval=0.5,
        relu_iterations=100,
        vram_to_keep=vram_elements,
        busy_threshold=10,
    )

    try:
        controller.keep()
        time.sleep(2)  # Give thread time to start and allocate
        assert controller._thread is not None and controller._thread.is_alive()
    finally:
        controller.release()
