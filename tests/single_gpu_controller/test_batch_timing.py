import pytest

import keep_gpu.single_gpu_controller.cuda_gpu_controller as cuda_module
import keep_gpu.single_gpu_controller.macm_gpu_controller as macm_module
import keep_gpu.single_gpu_controller.rocm_gpu_controller as rocm_module
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController
from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


def _forbid_wall_clock():
    raise AssertionError("batch elapsed timing must not use wall-clock time")


def _monotonic_sequence(*values):
    values_iter = iter(values)
    return lambda: next(values_iter)


def test_cuda_relu_batch_uses_monotonic_elapsed_time(monkeypatch):
    ctrl = CudaGPUController.__new__(CudaGPUController)
    ctrl.rank = 0
    ctrl.relu_iterations = 2
    ctrl._stop_evt = None
    relu_calls = []
    logs = []

    monkeypatch.setattr(cuda_module.time, "time", _forbid_wall_clock)
    monkeypatch.setattr(
        cuda_module.time, "monotonic", _monotonic_sequence(10.0, 10.006)
    )
    monkeypatch.setattr(
        cuda_module.torch, "relu_", lambda tensor: relu_calls.append(tensor)
    )
    monkeypatch.setattr(cuda_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        cuda_module.logger, "debug", lambda message, *args: logs.append((message, args))
    )

    ctrl._run_relu_batch(object())

    assert len(relu_calls) == 2
    assert logs == [
        (
            "rank %s: relu ops batch done - avg %.2f ms",
            (0, pytest.approx(3.0)),
        )
    ]


def test_rocm_batch_uses_monotonic_elapsed_time(monkeypatch):
    ctrl = RocmGPUController.__new__(RocmGPUController)
    ctrl.rank = 1
    ctrl.iterations = 3
    ctrl._stop_evt = None
    relu_calls = []
    logs = []

    monkeypatch.setattr(rocm_module.time, "time", _forbid_wall_clock)
    monkeypatch.setattr(
        rocm_module.time, "monotonic", _monotonic_sequence(20.0, 20.012)
    )
    monkeypatch.setattr(
        rocm_module.torch, "relu_", lambda tensor: relu_calls.append(tensor)
    )
    monkeypatch.setattr(rocm_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        rocm_module.logger, "debug", lambda message, *args: logs.append((message, args))
    )

    ctrl._run_batch(object())

    assert len(relu_calls) == 3
    assert logs == [
        (
            "rank %s: elementwise batch done - avg %.2f ms",
            (1, pytest.approx(4.0)),
        )
    ]


def test_macm_batch_uses_monotonic_elapsed_time(monkeypatch):
    ctrl = MacMGPUController.__new__(MacMGPUController)
    ctrl.rank = 2
    ctrl.iterations = 4
    ctrl._stop_evt = None
    relu_calls = []
    logs = []

    monkeypatch.setattr(macm_module.time, "time", _forbid_wall_clock)
    monkeypatch.setattr(
        macm_module.time, "monotonic", _monotonic_sequence(30.0, 30.020)
    )
    monkeypatch.setattr(
        macm_module.torch, "relu_", lambda tensor: relu_calls.append(tensor)
    )
    monkeypatch.setattr(macm_module.torch.mps, "synchronize", lambda: None)
    monkeypatch.setattr(
        macm_module.logger, "debug", lambda message, *args: logs.append((message, args))
    )

    ctrl._run_batch(object())

    assert len(relu_calls) == 4
    assert logs == [
        (
            "rank %s: elementwise batch done - avg %.2f ms",
            (2, pytest.approx(5.0)),
        )
    ]
