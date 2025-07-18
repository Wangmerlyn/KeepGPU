import time
import pytest
import torch
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.large_memory
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_large_vram_allocation():
    """Tests controller with a large VRAM allocation."""
    # Using a smaller allocation for general testing. The original 2**32 can be used on machines with sufficient VRAM.
    # torch has some indexing issues on very large tensors
    # e.g. tensors with more than 2**32-1 elements may cause issues
    # just a test to see if it is real.
    vram_elements = 2**32  # Allocates 16GiB to test large tensor handling
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
        assert controller._thread is not None and controller._thread.is_alive()
    finally:
        controller.release()


if __name__ == "__main__":
    test_large_vram_allocation()
