import threading

import pytest

from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
from keep_gpu.single_gpu_controller.macm_gpu_controller import MacMGPUController
from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


class StuckThread:
    def __init__(self):
        self.join_timeout = None

    @staticmethod
    def is_alive():
        return True

    def join(self, timeout=None):
        self.join_timeout = timeout


class ControllableThread:
    def __init__(self):
        self.alive = True
        self.join_timeout = None

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_timeout = timeout


class StopsDuringJoinThread(ControllableThread):
    def join(self, timeout=None):
        self.join_timeout = timeout
        self.alive = False


@pytest.mark.parametrize(
    ("controller_cls", "extra_attrs"),
    [
        (CudaGPUController, {}),
        (RocmGPUController, {"_rocm_smi": None}),
        (MacMGPUController, {}),
    ],
)
def test_release_raises_timeout_when_worker_thread_survives(
    controller_cls, extra_attrs
):
    controller = controller_cls.__new__(controller_cls)
    controller.rank = 0
    controller.interval = 0.01
    controller._thread = StuckThread()
    controller._stop_evt = threading.Event()
    for name, value in extra_attrs.items():
        setattr(controller, name, value)

    with pytest.raises(TimeoutError, match="did not stop"):
        controller.release()

    assert controller._stop_evt.is_set()
    assert controller._thread.join_timeout >= 2.0


def test_rocm_release_without_worker_does_not_log_stopped(monkeypatch):
    controller = RocmGPUController.__new__(RocmGPUController)
    controller.rank = 0
    controller._thread = None
    controller._rocm_smi = None
    info_messages = []

    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.rocm_gpu_controller.logger.info",
        lambda message, *args: info_messages.append(message % args),
    )

    controller.release()

    assert info_messages == []


@pytest.mark.parametrize(
    ("controller_cls", "cache_path", "extra_attrs"),
    [
        (
            CudaGPUController,
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.torch.cuda.empty_cache",
            {},
        ),
        (
            RocmGPUController,
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.torch.cuda.empty_cache",
            {"_rocm_smi": None},
        ),
        (
            MacMGPUController,
            "keep_gpu.single_gpu_controller.macm_gpu_controller.torch.mps.empty_cache",
            {},
        ),
    ],
)
def test_release_cleans_cache_after_timed_out_worker_later_exits(
    monkeypatch, controller_cls, cache_path, extra_attrs
):
    controller = controller_cls.__new__(controller_cls)
    controller.rank = 0
    controller.interval = 0.01
    thread = ControllableThread()
    controller._thread = thread
    controller._stop_evt = threading.Event()
    for name, value in extra_attrs.items():
        setattr(controller, name, value)

    cache_calls = []
    monkeypatch.setattr(cache_path, lambda: cache_calls.append("empty_cache"))
    if controller_cls is MacMGPUController:
        monkeypatch.setattr(
            "keep_gpu.single_gpu_controller.macm_gpu_controller.gc.collect",
            lambda: cache_calls.append("gc.collect"),
        )

    with pytest.raises(TimeoutError, match="did not stop"):
        controller.release()

    assert cache_calls == []
    assert controller._stop_evt.is_set()
    assert controller._thread is thread

    thread.alive = False
    controller.release()

    expected_calls = (
        ["empty_cache", "gc.collect"]
        if controller_cls is MacMGPUController
        else ["empty_cache"]
    )
    assert cache_calls == expected_calls
    assert controller._thread is None
    assert controller._stop_evt is None


@pytest.mark.parametrize("stop_evt", [None, threading.Event()])
@pytest.mark.parametrize(
    ("controller_cls", "cache_path", "extra_attrs"),
    [
        (
            CudaGPUController,
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.torch.cuda.empty_cache",
            {},
        ),
        (
            RocmGPUController,
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.torch.cuda.empty_cache",
            {"_rocm_smi": None},
        ),
        (
            MacMGPUController,
            "keep_gpu.single_gpu_controller.macm_gpu_controller.torch.mps.empty_cache",
            {},
        ),
    ],
)
def test_release_cleans_dead_runtime_failed_worker(
    monkeypatch, controller_cls, cache_path, extra_attrs, stop_evt
):
    controller = controller_cls.__new__(controller_cls)
    controller.rank = 0
    controller.interval = 0.01
    thread = ControllableThread()
    thread.alive = False
    controller._thread = thread
    controller._stop_evt = stop_evt
    controller._failure_exc = RuntimeError("worker failed")
    for name, value in extra_attrs.items():
        setattr(controller, name, value)

    cache_calls = []
    monkeypatch.setattr(cache_path, lambda: cache_calls.append("empty_cache"))
    if controller_cls is MacMGPUController:
        monkeypatch.setattr(
            "keep_gpu.single_gpu_controller.macm_gpu_controller.gc.collect",
            lambda: cache_calls.append("gc.collect"),
        )

    controller.release()

    expected_calls = (
        ["empty_cache", "gc.collect"]
        if controller_cls is MacMGPUController
        else ["empty_cache"]
    )
    assert cache_calls == expected_calls
    assert controller._thread is None
    assert controller._stop_evt is None


@pytest.mark.parametrize(
    ("controller_cls", "cache_path", "logger_path", "extra_attrs"),
    [
        (
            CudaGPUController,
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.torch.cuda.empty_cache",
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.logger.warning",
            {},
        ),
        (
            RocmGPUController,
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.torch.cuda.empty_cache",
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.logger.warning",
            {"_rocm_smi": None},
        ),
        (
            MacMGPUController,
            "keep_gpu.single_gpu_controller.macm_gpu_controller.torch.mps.empty_cache",
            "keep_gpu.single_gpu_controller.macm_gpu_controller.logger.warning",
            {},
        ),
    ],
)
def test_release_success_clears_state_so_second_release_keeps_not_running_behavior(
    monkeypatch, controller_cls, cache_path, logger_path, extra_attrs
):
    controller = controller_cls.__new__(controller_cls)
    controller.rank = 0
    controller.interval = 0.01
    thread = StopsDuringJoinThread()
    controller._thread = thread
    controller._stop_evt = threading.Event()
    for name, value in extra_attrs.items():
        setattr(controller, name, value)

    cache_calls = []
    warnings = []
    monkeypatch.setattr(cache_path, lambda: cache_calls.append("empty_cache"))
    if controller_cls is MacMGPUController:
        monkeypatch.setattr(
            "keep_gpu.single_gpu_controller.macm_gpu_controller.gc.collect",
            lambda: cache_calls.append("gc.collect"),
        )
    monkeypatch.setattr(
        logger_path,
        lambda message, *args: warnings.append(message % args),
    )

    controller.release()

    expected_calls = (
        ["empty_cache", "gc.collect"]
        if controller_cls is MacMGPUController
        else ["empty_cache"]
    )
    assert cache_calls == expected_calls
    assert controller._thread is None
    assert controller._stop_evt is None

    controller.release()

    assert cache_calls == expected_calls
    assert warnings == ["rank 0: keep thread not running"]


@pytest.mark.parametrize("stop_evt", [None, threading.Event()])
@pytest.mark.parametrize(
    ("controller_cls", "cache_path", "logger_path", "extra_attrs"),
    [
        (
            CudaGPUController,
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.torch.cuda.empty_cache",
            "keep_gpu.single_gpu_controller.cuda_gpu_controller.logger.warning",
            {},
        ),
        (
            RocmGPUController,
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.torch.cuda.empty_cache",
            "keep_gpu.single_gpu_controller.rocm_gpu_controller.logger.warning",
            {"_rocm_smi": None},
        ),
        (
            MacMGPUController,
            "keep_gpu.single_gpu_controller.macm_gpu_controller.torch.mps.empty_cache",
            "keep_gpu.single_gpu_controller.macm_gpu_controller.logger.warning",
            {},
        ),
    ],
)
def test_release_dead_thread_without_stopping_state_keeps_not_running_behavior(
    monkeypatch, controller_cls, cache_path, logger_path, extra_attrs, stop_evt
):
    controller = controller_cls.__new__(controller_cls)
    controller.rank = 0
    controller.interval = 0.01
    thread = ControllableThread()
    thread.alive = False
    controller._thread = thread
    controller._stop_evt = stop_evt
    for name, value in extra_attrs.items():
        setattr(controller, name, value)

    cache_calls = []
    warnings = []
    monkeypatch.setattr(cache_path, lambda: cache_calls.append("empty_cache"))
    monkeypatch.setattr(
        logger_path,
        lambda message, *args: warnings.append(message % args),
    )

    controller.release()

    assert cache_calls == []
    assert warnings == ["rank 0: keep thread not running"]
    assert controller._thread is thread
    assert controller._stop_evt is stop_evt


def test_rocm_late_release_still_shuts_down_rocm_smi(monkeypatch):
    class FakeRocmSmi:
        def __init__(self):
            self.shutdown_calls = 0

        def rsmi_shut_down(self):
            self.shutdown_calls += 1

    controller = RocmGPUController.__new__(RocmGPUController)
    controller.rank = 0
    controller.interval = 0.01
    thread = ControllableThread()
    controller._thread = thread
    controller._stop_evt = threading.Event()
    rocm_smi = FakeRocmSmi()
    controller._rocm_smi = rocm_smi
    monkeypatch.setattr(
        "keep_gpu.single_gpu_controller.rocm_gpu_controller.torch.cuda.empty_cache",
        lambda: None,
    )

    with pytest.raises(TimeoutError, match="did not stop"):
        controller.release()

    thread.alive = False
    controller.release()

    assert rocm_smi.shutdown_calls == 2
