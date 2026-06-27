import time

import pytest
import torch

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities import platform_manager as pm


def test_global_keep_rolls_back_started_controllers(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    instances = []

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            self.rank = rank
            self.kept = False
            self.released = False
            instances.append(self)

        def keep(self):
            if self.rank == 1:
                raise RuntimeError("rank 1 failed to start")
            self.kept = True

        def release(self):
            self.released = True

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    controller = GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="8MB")

    with pytest.raises(RuntimeError, match="rank 1 failed to start"):
        controller.keep()

    assert instances[0].kept is True
    assert instances[0].released is True
    assert instances[1].kept is False
    assert instances[1].released is False


def test_global_controller_rejects_zero_visible_cuda_devices(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    instances = []

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            instances.append(self)

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    with pytest.raises(ValueError, match="No GPUs available for GlobalGPUController"):
        GlobalGPUController(gpu_ids=None, vram_to_keep="8MB")

    assert instances == []


def test_global_controller_rejects_explicit_cuda_id_outside_visible_count(
    monkeypatch,
):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    instances = []

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            instances.append(self)

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    with pytest.raises(
        ValueError,
        match="gpu_ids must be visible device ordinals less than 1",
    ):
        GlobalGPUController(gpu_ids=[1], vram_to_keep="8MB")

    assert instances == []


def test_global_controller_rejects_explicit_rocm_id_outside_visible_count(
    monkeypatch,
):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.ROCM)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    instances = []

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            instances.append(self)

    import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module

    monkeypatch.setattr(rocm_module, "RocmGPUController", DummyController)

    with pytest.raises(
        ValueError,
        match="gpu_ids must be visible device ordinals less than 2",
    ):
        GlobalGPUController(gpu_ids=[2], vram_to_keep="8MB")

    assert instances == []


def test_global_release_attempts_all_controllers_and_reports_failures():
    class DummyController:
        def __init__(self, rank, error=None):
            self.rank = rank
            self.error = error
            self.released = False

        def release(self):
            self.released = True
            if self.error:
                raise self.error

    controllers = [
        DummyController(0),
        DummyController(1, RuntimeError("release failed")),
        DummyController(2),
    ]
    controller = GlobalGPUController.__new__(GlobalGPUController)
    controller.controllers = controllers

    with pytest.raises(RuntimeError, match="rank 1: release failed"):
        controller.release()

    assert [ctrl.released for ctrl in controllers] == [True, True, True]


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
