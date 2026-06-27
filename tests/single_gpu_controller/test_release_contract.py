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
