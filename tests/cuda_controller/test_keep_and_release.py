import time
import torch
import pytest
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_basic():
    ctrl = CudaGPUController(
        rank=torch.cuda.device_count() - 1, interval=10, vram_to_keep="100MB"
    )
    ctrl.keep()
    print("GPU kept busy for 10 seconds.")

    time.sleep(10)
    ctrl.release()
    print("GPU released.")

    print("test for 2nd time")
    ctrl.keep()
    print("GPU kept busy for another 10 seconds.")
    time.sleep(10)
    ctrl.release()
    print("GPU released again.")
    print("test for 3rd time")
    with ctrl:
        print("GPU kept busy in context manager for 10 seconds.")
        time.sleep(10)
    print("GPU released after context manager.")
    print("Test completed successfully.")
    # This code snippet is for testing the CudaGPUController functionality.


if __name__ == "__main__":
    test_cuda_controller_basic()
