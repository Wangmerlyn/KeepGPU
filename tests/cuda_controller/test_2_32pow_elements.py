import time
import pytest
import torch
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.large_memory
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_vram_allocation():
    """Tests controller with a large VRAM allocation."""
    # Using a smaller allocation for general testing. The original 2**32 can be used on machines with sufficient VRAM.
    vram_elements = 2**28  # Allocates 1GB, more reasonable for a standard test
    controller = CudaGPUController(
        rank=0,
        interval=0.5,
        matmul_iterations=100,
        vram_to_keep=vram_elements,
        busy_threshold=10,
    )

    try:
        controller.keep()
        time.sleep(2)  # Give thread time to start and allocate
        # A full test would assert on thread status or logs.
    finally:
        controller.release()
