import types

from keep_gpu.utilities.gpu_monitor import NVMLMonitor


class DummyNVML:
    """Minimal stand-in for the NVML module used in tests."""

    class NVMLError(Exception):
        pass

    def __init__(self, should_fail: bool = False, gpu_util: int = 50) -> None:
        self.should_fail = should_fail
        self.gpu_util = gpu_util
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1
        if self.should_fail:
            raise self.NVMLError("init failure")

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index: int):
        if self.should_fail:
            raise self.NVMLError("handle failure")
        return types.SimpleNamespace(index=index)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return types.SimpleNamespace(gpu=self.gpu_util)


def test_monitor_returns_none_when_nvml_missing():
    monitor = NVMLMonitor(None)
    assert monitor.get_gpu_utilization(0) is None


def test_monitor_reads_gpu_utilization():
    dummy = DummyNVML(gpu_util=73)
    monitor = NVMLMonitor(dummy)
    assert monitor.get_gpu_utilization(1) == 73
    # second call reuses initialization
    assert monitor.get_gpu_utilization(2) == 73
    assert dummy.init_calls == 1


def test_monitor_handles_nvml_errors():
    dummy = DummyNVML(should_fail=True)
    monitor = NVMLMonitor(dummy)
    assert monitor.get_gpu_utilization(0) is None
