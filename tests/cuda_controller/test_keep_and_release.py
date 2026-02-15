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
        rank=0,
        interval=0.05,
        vram_to_keep="8MB",
        matmul_iterations=64,
    )
    ctrl.keep()
    time.sleep(0.2)
    assert ctrl._thread and ctrl._thread.is_alive()

    assert ctrl._thread is not None
    ctrl.release()
    ctrl._thread.join(timeout=2)
    assert not (ctrl._thread and ctrl._thread.is_alive())

    ctrl.keep()
    time.sleep(0.2)
    assert ctrl._thread and ctrl._thread.is_alive()
    assert ctrl._thread is not None
    ctrl.release()
    ctrl._thread.join(timeout=2)
    assert not (ctrl._thread and ctrl._thread.is_alive())

    with ctrl:
        assert ctrl._thread and ctrl._thread.is_alive()
        time.sleep(0.2)
    assert ctrl._thread is not None
    ctrl._thread.join(timeout=2)
    assert not (ctrl._thread and ctrl._thread.is_alive())
