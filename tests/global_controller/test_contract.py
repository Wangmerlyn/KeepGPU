import pytest

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities import platform_manager as pm


def test_global_controller_rejects_busy_threshold_below_minus_one(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", pm.ComputingPlatform.CPU)

    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        GlobalGPUController(busy_threshold=-2)
