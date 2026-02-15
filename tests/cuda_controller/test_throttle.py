import time

import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Only run CUDA tests when CUDA is available",
)
def test_controller_respects_busy_threshold(monkeypatch):
    calls = {"run": 0}

    ctrl = CudaGPUController(
        rank=0,
        interval=0.05,
        vram_to_keep="4MB",
        busy_threshold=10,
        relu_iterations=8,
    )

    def fake_utilization(_rank: int) -> int:
        return 100  # always above threshold

    def fake_run(matrix):
        calls["run"] += 1

    monkeypatch.setattr(ctrl, "_monitor_utilization", fake_utilization)
    monkeypatch.setattr(ctrl, "_run_relu_batch", fake_run)

    ctrl.keep()
    time.sleep(0.2)
    ctrl.release()

    # When utilization is always high, no matmul batches should run
    assert calls["run"] == 0
