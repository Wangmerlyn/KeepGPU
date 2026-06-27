import types

from keep_gpu.utilities.gpu_monitor import NVMLMonitor


class DummyNVML:
    """Minimal stand-in for the NVML module used in tests."""

    class NVMLError(Exception):
        pass

    def __init__(
        self,
        should_fail: bool = False,
        gpu_util: int = 50,
        util_by_index: dict[int, int] | None = None,
        util_by_uuid: dict[str, int] | None = None,
        fail_uuid_lookup: bool = False,
        support_uuid_lookup: bool = False,
        invalid_indexes: set[int] | None = None,
    ) -> None:
        self.should_fail = should_fail
        self.gpu_util = gpu_util
        self.init_calls = 0
        self.queried_indexes: list[int] = []
        self.queried_uuids: list[str] = []
        self.util_by_index = util_by_index or {}
        self.util_by_uuid = util_by_uuid or {}
        self.fail_uuid_lookup = fail_uuid_lookup
        self.invalid_indexes = invalid_indexes or set()
        if support_uuid_lookup:
            self.nvmlDeviceGetHandleByUUID = self._nvmlDeviceGetHandleByUUID

    def nvmlInit(self):
        self.init_calls += 1
        if self.should_fail:
            raise self.NVMLError("init failure")

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index: int):
        if self.should_fail:
            raise self.NVMLError("handle failure")
        if index in self.invalid_indexes:
            raise self.NVMLError("invalid index")
        self.queried_indexes.append(index)
        return types.SimpleNamespace(index=index)

    def _nvmlDeviceGetHandleByUUID(self, uuid: str):
        self.queried_uuids.append(uuid)
        if self.fail_uuid_lookup:
            raise self.NVMLError("uuid lookup failure")
        return types.SimpleNamespace(uuid=uuid)

    def nvmlDeviceGetUtilizationRates(self, handle):
        if hasattr(handle, "uuid"):
            return types.SimpleNamespace(
                gpu=self.util_by_uuid.get(handle.uuid, self.gpu_util)
            )
        return types.SimpleNamespace(
            gpu=self.util_by_index.get(handle.index, self.gpu_util)
        )


def test_monitor_returns_none_when_nvml_missing():
    monitor = NVMLMonitor(None)
    assert monitor.get_gpu_utilization(0) is None


def test_monitor_reads_gpu_utilization(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyNVML(gpu_util=73)
    monitor = NVMLMonitor(dummy)
    assert monitor.get_gpu_utilization(1) == 73
    # second call reuses initialization
    assert monitor.get_gpu_utilization(2) == 73
    assert dummy.init_calls == 1
    assert dummy.queried_indexes == [1, 2]


def test_monitor_handles_nvml_errors(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyNVML(should_fail=True)
    monitor = NVMLMonitor(dummy)
    assert monitor.get_gpu_utilization(0) is None


def test_monitor_maps_visible_ordinal_through_numeric_cuda_visible_devices(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3, 5")
    dummy = DummyNVML(util_by_index={5: 85})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) == 85
    assert dummy.queried_indexes == [3, 5]
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_out_of_range_visible_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(2) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_prior_token_is_invalid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1,2")
    dummy = DummyNVML(util_by_index={2: 99})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(2) is None
    assert dummy.queried_indexes == [0]
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_prior_token_is_empty(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,,2")
    dummy = DummyNVML(util_by_index={2: 99})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(2) is None
    assert dummy.queried_indexes == [0]
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_prior_numeric_index_is_invalid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,2")
    dummy = DummyNVML(util_by_index={2: 99}, invalid_indexes={99})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(2) is None
    assert dummy.queried_indexes == [0]
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_empty_cuda_visible_devices_token(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,,5")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == [3]
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_uuid_token_without_lookup_support(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_queries_uuid_token_when_nvml_supports_uuid_lookup(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(
        gpu_util=99,
        util_by_uuid={"GPU-abc123": 42},
        support_uuid_lookup=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) == 42
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123"]


def test_monitor_returns_none_when_uuid_lookup_fails(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(fail_uuid_lookup=True, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123"]


def test_monitor_returns_none_when_prior_uuid_lookup_fails(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123,2")
    dummy = DummyNVML(
        util_by_index={2: 99},
        fail_uuid_lookup=True,
        support_uuid_lookup=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123"]
