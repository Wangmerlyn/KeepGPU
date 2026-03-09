import sys
import time

import pytest
import torch

from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController
from keep_gpu.utilities.platform_manager import ComputingPlatform


pytestmark = [
    pytest.mark.skipif(
        not (sys.platform == "darwin" and torch.backends.mps.is_available()),
        reason="Only run MacM tests on Apple Silicon with MPS",
    ),
    pytest.mark.macm,
]


def test_macm_controller_basic():
    controller = MacMGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="8MB",
        iterations=64,
    )

    controller.keep()
    time.sleep(0.2)
    assert controller._thread and controller._thread.is_alive()

    controller.release()
    assert not (controller._thread and controller._thread.is_alive())

    controller.keep()
    time.sleep(0.2)
    assert controller._thread and controller._thread.is_alive()

    controller.release()
    assert not (controller._thread and controller._thread.is_alive())


def test_macm_controller_context_manager():
    with MacMGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="8MB",
        iterations=64,
    ) as controller:
        time.sleep(0.2)
        assert controller._thread and controller._thread.is_alive()

    assert not (controller._thread and controller._thread.is_alive())


def test_macm_controller_invalid_rank():
    with pytest.raises(ValueError, match="MPS only supports device 0"):
        MacMGPUController(
            rank=1,
            interval=0.05,
            vram_to_keep="8MB",
            iterations=64,
        )


def test_macm_controller_platform():
    controller = MacMGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="8MB",
        iterations=64,
    )
    assert controller.platform == ComputingPlatform.MACM
