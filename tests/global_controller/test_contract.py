import inspect

import pytest

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController
from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController
from keep_gpu.utilities import platform_manager as pm


def test_python_controller_defaults_use_eco_safe_busy_threshold():
    controllers = [
        GlobalGPUController,
        CudaGPUController,
        RocmGPUController,
        MacMGPUController,
    ]

    for controller in controllers:
        assert inspect.signature(controller).parameters["busy_threshold"].default == 25


def test_global_controller_rejects_busy_threshold_below_minus_one(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CPU)

    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        GlobalGPUController(busy_threshold=-2)


def test_global_controller_rejects_empty_gpu_ids(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CPU)

    with pytest.raises(ValueError, match="gpu_ids must select at least one GPU"):
        GlobalGPUController(gpu_ids=[])
