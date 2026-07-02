import time

import pytest
import torch

from keep_gpu.global_gpu_controller.global_gpu_controller import (
    GlobalGPUController,
    NoGPUAvailableError,
)
from keep_gpu.utilities import platform_manager as pm


def test_global_controller_propagates_default_vram_to_child_controller(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    captured = {}

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            captured["vram_to_keep"] = vram_to_keep

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    GlobalGPUController(gpu_ids=[0])

    assert captured["vram_to_keep"] == "1GiB"


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


def test_global_keep_does_not_roll_back_already_running_controller(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    instances = []

    class RunningThread:
        @staticmethod
        def is_alive():
            return True

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            self.rank = rank
            self.keep_calls = 0
            self.released = False
            self._thread = RunningThread() if rank == 0 else None
            self._stop_evt = object() if rank == 0 else None
            instances.append(self)

        def keep(self):
            self.keep_calls += 1
            if self.rank == 1:
                raise RuntimeError("rank 1 failed to start")

        def release(self):
            self.released = True

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    controller = GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="8MB")

    with pytest.raises(RuntimeError, match="rank 1 failed to start"):
        controller.keep()

    assert instances[0].keep_calls == 1
    assert instances[0].released is False
    assert instances[1].released is False


def test_global_keep_releases_failed_controller_with_worker_state(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    instances = []

    class DummyThread:
        @staticmethod
        def is_alive():
            return True

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            self.rank = rank
            self.kept = False
            self.released = False
            self._thread = None
            self._stop_evt = None
            instances.append(self)

        def keep(self):
            if self.rank == 1:
                self._thread = DummyThread()
                self._stop_evt = object()
                raise RuntimeError("rank 1 failed after spawning worker")
            self.kept = True

        def release(self):
            self.released = True

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    controller = GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="8MB")

    with pytest.raises(RuntimeError, match="rank 1 failed after spawning worker"):
        controller.keep()

    assert instances[0].released is True
    assert instances[1].released is True


def test_global_keep_preserves_start_error_when_failed_child_cleanup_fails(
    monkeypatch,
):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CUDA)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    instances = []

    class DummyThread:
        @staticmethod
        def is_alive():
            return True

    class DummyController:
        def __init__(self, *, rank, interval, vram_to_keep, busy_threshold):
            self.rank = rank
            self.released = False
            self._thread = None
            self._stop_evt = None
            instances.append(self)

        def keep(self):
            if self.rank == 1:
                self._thread = DummyThread()
                self._stop_evt = object()
                raise RuntimeError("original start failure")

        def release(self):
            self.released = True
            if self.rank == 1:
                raise RuntimeError("cleanup failed")

    import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module

    monkeypatch.setattr(cuda_module, "CudaGPUController", DummyController)

    controller = GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="8MB")

    with pytest.raises(RuntimeError, match="original start failure"):
        controller.keep()

    assert instances[0].released is True
    assert instances[1].released is True


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


def test_global_controller_reports_unavailable_mps_as_startup_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        NoGPUAvailableError,
        match="PyTorch MPS backend is not available",
    ):
        GlobalGPUController(gpu_ids=[0], vram_to_keep=4)


def test_global_controller_reports_mps_probe_exception_as_startup_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.MACM)

    import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module

    def raise_probe_error():
        raise ValueError("MPS probe exploded")

    monkeypatch.setattr(
        macm_module.torch.backends.mps,
        "is_available",
        raise_probe_error,
    )

    with pytest.raises(
        NoGPUAvailableError,
        match="PyTorch MPS backend availability check failed: MPS probe exploded",
    ):
        GlobalGPUController(gpu_ids=[0], vram_to_keep=4)


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


def test_global_runtime_error_reports_first_child_allocation_failure():
    class DummyController:
        def __init__(self, rank, error=None):
            self.rank = rank
            self.error = error

        def allocation_status(self):
            return self.error

    controller = GlobalGPUController.__new__(GlobalGPUController)
    controller.controllers = [
        DummyController(0),
        DummyController(1, RuntimeError("allocation retries exhausted")),
        DummyController(2, RuntimeError("later failure")),
    ]

    error = controller.runtime_error()

    assert isinstance(error, RuntimeError)
    assert str(error) == "rank 1: allocation retries exhausted"


def test_global_runtime_error_preserves_rank_prefixed_child_failure():
    child_error = RuntimeError(
        "rank 0: unexpected CUDA keep worker failure: device-side assert"
    )

    class DummyController:
        rank = 0

        @staticmethod
        def allocation_status():
            return child_error

    controller = GlobalGPUController.__new__(GlobalGPUController)
    controller.controllers = [DummyController()]

    error = controller.runtime_error()

    assert error is child_error
    assert (
        str(error) == "rank 0: unexpected CUDA keep worker failure: device-side assert"
    )


def test_global_runtime_error_ignores_healthy_or_unreported_controllers():
    class ControllerWithoutHealthHook:
        rank = 0

    class HealthyController:
        rank = 1

        @staticmethod
        def allocation_status():
            return None

    controller = GlobalGPUController.__new__(GlobalGPUController)
    controller.controllers = [ControllerWithoutHealthHook(), HealthyController()]

    assert controller.runtime_error() is None


def test_global_runtime_error_wraps_health_hook_exceptions():
    class FailingHealthController:
        rank = 0

        @staticmethod
        def allocation_status():
            raise RuntimeError("health probe exploded")

    controller = GlobalGPUController.__new__(GlobalGPUController)
    controller.controllers = [FailingHealthController()]

    error = controller.runtime_error()

    assert isinstance(error, RuntimeError)
    assert str(error) == "rank 0: health probe exploded"


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
