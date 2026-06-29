import inspect
import math

import pytest
import torch

from keep_gpu.global_gpu_controller import global_gpu_controller as global_module
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


@pytest.mark.parametrize(
    "controller",
    [
        GlobalGPUController,
        CudaGPUController,
        RocmGPUController,
        MacMGPUController,
    ],
)
def test_python_controller_default_vram_matches_public_low_power_default(controller):
    default = inspect.signature(controller).parameters["vram_to_keep"].default
    assert default == "1GiB"


@pytest.mark.parametrize("iterations", [1.5, True, "5000"])
@pytest.mark.parametrize(
    ("controller", "parameter", "message"),
    [
        (CudaGPUController, "relu_iterations", "relu_iterations must be an integer"),
        (RocmGPUController, "iterations", "iterations must be an integer"),
        (MacMGPUController, "iterations", "iterations must be an integer"),
    ],
)
def test_single_gpu_controllers_reject_non_integer_workload_iterations(
    controller, parameter, message, iterations
):
    with pytest.raises(TypeError, match=message):
        controller(rank=0, vram_to_keep=4, **{parameter: iterations})


@pytest.mark.parametrize("rank", [1.5, True, "0"])
@pytest.mark.parametrize(
    "controller",
    [
        CudaGPUController,
        RocmGPUController,
    ],
)
def test_direct_cuda_rocm_controllers_reject_non_integer_ranks(
    monkeypatch, controller, rank
):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(TypeError, match="rank must be an integer"):
        controller(rank=rank, vram_to_keep=4)


@pytest.mark.parametrize("rank", [1.5, True, "0"])
@pytest.mark.parametrize(
    "controller",
    [
        CudaGPUController,
        RocmGPUController,
    ],
)
def test_direct_cuda_rocm_controllers_reject_non_integer_ranks_before_device_count(
    monkeypatch, controller, rank
):
    def fail_device_count():
        raise AssertionError("device_count should not run for invalid rank type")

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_count)

    with pytest.raises(TypeError, match="rank must be an integer"):
        controller(rank=rank, vram_to_keep=4)


@pytest.mark.parametrize("rank", [-1, 1])
@pytest.mark.parametrize(
    "controller",
    [
        CudaGPUController,
        RocmGPUController,
    ],
)
def test_direct_cuda_rocm_controllers_reject_invalid_visible_ranks(
    monkeypatch, controller, rank
):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(ValueError, match="rank must be a visible device ordinal"):
        controller(rank=rank, vram_to_keep=4)


@pytest.mark.parametrize("matmul_iterations", [1.5, True, "5000"])
def test_cuda_controller_rejects_non_integer_matmul_iteration_alias(
    matmul_iterations,
):
    with pytest.raises(TypeError, match="relu_iterations must be an integer"):
        CudaGPUController(
            rank=0,
            vram_to_keep=4,
            matmul_iterations=matmul_iterations,
        )


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu_ids": []}, "gpu_ids must select at least one GPU"),
        ({"gpu_ids": [0, 0]}, "gpu_ids must not contain duplicate values"),
        ({"gpu_ids": [-1]}, "gpu_ids must contain non-negative integers"),
        ({"interval": 0}, "interval must be positive"),
        ({"interval": math.nan}, "interval must be finite and positive"),
        ({"interval": 10**1000}, "interval must be no more than"),
        (
            {"busy_threshold": -2},
            "busy_threshold must be -1 or an integer between 0 and 100",
        ),
        (
            {"busy_threshold": 101},
            "busy_threshold must be -1 or an integer between 0 and 100",
        ),
        ({"vram_to_keep": []}, "vram_to_keep must be str or int bytes"),
        ({"vram_to_keep": "not-a-size"}, "invalid format"),
        ({"vram_to_keep": 1}, "memory size must be at least 4 bytes"),
        ({"vram_to_keep": 10**1000}, "vram must be no more than"),
    ],
)
def test_global_controller_validates_local_inputs_before_platform_probe(
    monkeypatch, kwargs, message
):
    def fail_platform_probe():
        raise AssertionError("platform probe should not run for invalid inputs")

    monkeypatch.setattr(global_module, "get_platform", fail_platform_probe)

    with pytest.raises((TypeError, ValueError), match=message):
        GlobalGPUController(**kwargs)
