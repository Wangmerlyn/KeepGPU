import time
import pytest
import torch
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_global_controller():
    controller = GlobalGPUController(interval=10, vram_to_keep=2000)
    controller.keep()

    time.sleep(10)
    controller.release()
    print("done")
