import time
import pytest
import torch
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities import platform_manager as pm


def test_global_controller_accepts_fractional_interval(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            self.rank = rank
            self.interval = interval
            self.vram_to_keep = vram_to_keep
            self.busy_threshold = busy_threshold

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    controller = GlobalGPUController(
        gpu_ids=[0],
        interval=0.05,
        vram_to_keep="8MB",
        busy_threshold=10,
    )

    assert controller.interval == 0.05
    assert controller.controllers[0].interval == 0.05


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
