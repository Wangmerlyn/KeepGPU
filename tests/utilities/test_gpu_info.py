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


def test_get_gpu_info_rocm(monkeypatch):
    # remove nvml so ROCm path is used
    monkeypatch.setitem(sys.modules, "pynvml", None)

    class DummyTorchCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info():
            return (50, 100)

        @staticmethod
        def get_device_name(idx):
            return f"ROCm {idx}"

        @staticmethod
        def set_device(idx):
            return None

    monkeypatch.setattr(
        gpu_info,
        "torch",
        type(
            "T", (), {"cuda": DummyTorchCuda, "version": type("V", (), {"hip": "6.0"})}
        ),
    )

    class DummyROCM:
        calls = 0

        @classmethod
        def rsmi_init(cls):
            cls.calls += 1

        @classmethod
        def rsmi_dev_busy_percent_get(cls, idx):
            assert idx == 0
            return 77

        @classmethod
        def rsmi_shut_down(cls):
            cls.calls += 1

    monkeypatch.setitem(sys.modules, "rocm_smi", DummyROCM)

    infos = gpu_info.get_gpu_info()
    assert len(infos) == 1
    info = infos[0]
    assert info["platform"] == "rocm"
    assert info["utilization"] == 77
    assert info["memory_total"] == 100
    assert info["memory_used"] == 50
