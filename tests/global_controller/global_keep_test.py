import time
import pytest
import torch
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_global_controller():
    controller = GlobalGPUController(
        gpu_ids=[0],
        interval=0.05,
        vram_to_keep="8MB",
        busy_threshold=10,
    )
    controller.keep()

    time.sleep(0.3)
    for ctrl in controller.controllers:
        assert ctrl._thread and ctrl._thread.is_alive()
    controller.release()
    for ctrl in controller.controllers:
        assert not (ctrl._thread and ctrl._thread.is_alive())


if __name__ == "__main__":
    test_global_controller()
