import sys

from keep_gpu.utilities import gpu_info


class DummyNVMLMemory:
    def __init__(self, total: int, used: int):
        self.total = total
        self.used = used


class DummyNVMLUtil:
    def __init__(self, gpu: int):
        self.gpu = gpu


def test_get_gpu_info_nvml(monkeypatch):
    class DummyNVML:
        def __init__(self):
            self.shutdown_calls = 0

        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlDeviceGetCount():
            return 1

        @staticmethod
        def nvmlDeviceGetHandleByIndex(index):
            assert index == 0
            return "handle"

        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle):
            return DummyNVMLMemory(total=2048, used=1024)

        @staticmethod
        def nvmlDeviceGetUtilizationRates(handle):
            return DummyNVMLUtil(gpu=55)

        @staticmethod
        def nvmlDeviceGetName(handle):
            return b"Mock GPU"

        def nvmlShutdown(self):
            self.shutdown_calls += 1

    dummy_nvml = DummyNVML()
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)

    infos = gpu_info.get_gpu_info()
    assert len(infos) == 1
    info = infos[0]
    assert info["name"] == "Mock GPU"
    assert info["memory_total"] == 2048
    assert info["memory_used"] == 1024
    assert info["utilization"] == 55
