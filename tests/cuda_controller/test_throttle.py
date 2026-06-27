import time

import pytest
import torch

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController


def test_negative_busy_threshold_disables_backoff_without_gpu():
    assert CudaGPUController._should_run_batch(0, -1) is True
    assert CudaGPUController._should_run_batch(100, -1) is True
    assert CudaGPUController._should_run_batch(None, -1) is True


def test_non_negative_busy_threshold_backs_off_above_limit_without_gpu():
    assert CudaGPUController._should_run_batch(10, 10) is True
    assert CudaGPUController._should_run_batch(11, 10) is False
    assert CudaGPUController._should_run_batch(None, 10) is False


def test_cuda_monitor_preserves_unknown_utilization(monkeypatch):
    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.cuda_gpu_controller.get_gpu_utilization",
        lambda rank: None,
    )

    assert CudaGPUController._monitor_utilization(0) is None


def test_cuda_controller_rejects_busy_threshold_below_minus_one_without_gpu():
    with pytest.raises(ValueError, match="busy_threshold must be an integer >= -1"):
        CudaGPUController(rank=0, vram_to_keep="4MB", busy_threshold=-2)


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
