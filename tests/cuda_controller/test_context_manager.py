import time
import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_cuda_controller_context_manager():
    ctrl = CudaGPUController(
        rank=torch.cuda.device_count() - 1,
        interval=0.05,
        vram_to_keep="8MB",
        relu_iterations=64,
    )

    torch.cuda.set_device(ctrl.rank)
    before_reserved = torch.cuda.memory_reserved(ctrl.rank)
    with ctrl:
        time.sleep(0.3)
        assert ctrl._thread and ctrl._thread.is_alive()
        during_reserved = torch.cuda.memory_reserved(ctrl.rank)
        assert during_reserved >= before_reserved

    assert not (ctrl._thread and ctrl._thread.is_alive())
