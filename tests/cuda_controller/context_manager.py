import time
import torch
import pytest
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_context_manager():
    ctrl = CudaGPUController(
        rank=torch.cuda.device_count() - 1, interval=10, vram_to_keep="1GB"
    )

    with ctrl:
        print("GPU kept busy for 10 seconds.")
        time.sleep(10)
        print("GPU released.")
    print("Test completed successfully.")


if __name__ == "__main__":
    test_cuda_controller_context_manager()
