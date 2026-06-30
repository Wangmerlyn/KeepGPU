from __future__ import annotations

import types

import pytest

from keep_gpu.utilities.gpu_monitor import NVMLMonitor

OVERSIZED_NUMERIC_TOKEN = "9" * 100


class DummyNVML:
    """Minimal stand-in for the NVML module used in tests."""

    class NVMLError(Exception):
        pass

    def __init__(
        self,
        should_fail: bool = False,
        gpu_util: int = 50,
        util_by_index: dict[int, int] | None = None,
        util_by_uuid: dict[object, int] | None = None,
        fail_uuid_lookup: bool = False,
        uuid_string_raises_nvml: bool = False,
        uuid_requires_bytes: bool = False,
        uuid_always_type_error: bool = False,
        support_uuid_lookup: bool = False,
        invalid_indexes: set[int] | None = None,
        uuid_index_by_value: dict[object, int] | None = None,
        count: int = 8,
    ) -> None:
        self.should_fail = should_fail
        self.gpu_util = gpu_util
        self.count = count
        self.init_calls = 0
        self.shutdown_calls = 0
        self.initialized = False
        self.queried_indexes: list[int] = []
        self.queried_uuids: list[object] = []
        self.util_by_index = util_by_index or {}
        self.util_by_uuid = util_by_uuid or {}
        self.fail_uuid_lookup = fail_uuid_lookup
        self.uuid_string_raises_nvml = uuid_string_raises_nvml
        self.uuid_requires_bytes = uuid_requires_bytes
        self.uuid_always_type_error = uuid_always_type_error
        self.invalid_indexes = invalid_indexes or set()
        self.uuid_index_by_value = uuid_index_by_value or {}
        if support_uuid_lookup:
            self.nvmlDeviceGetHandleByUUID = self._nvmlDeviceGetHandleByUUID

    def nvmlInit(self):
        self.init_calls += 1
        if self.should_fail:
            raise self.NVMLError("init failure")
        self.initialized = True

    def nvmlShutdown(self):
        self.shutdown_calls += 1
        self.initialized = False

    def _require_initialized(self):
        if not self.initialized:
            raise self.NVMLError("NVML is not initialized")

    def nvmlDeviceGetCount(self):
        self._require_initialized()
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index: int):
        self._require_initialized()
        if self.should_fail:
            raise self.NVMLError("handle failure")
        if index in self.invalid_indexes:
            raise self.NVMLError("invalid index")
        self.queried_indexes.append(index)
        return types.SimpleNamespace(index=index)

    def _nvmlDeviceGetHandleByUUID(self, uuid: str):
        self._require_initialized()
        self.queried_uuids.append(uuid)
        if self.uuid_always_type_error or (
            self.uuid_requires_bytes and isinstance(uuid, str)
        ):
            raise TypeError("uuid must be bytes")
        if self.uuid_string_raises_nvml and isinstance(uuid, str):
            raise self.NVMLError("uuid string lookup failure")
        if self.fail_uuid_lookup:
            raise self.NVMLError("uuid lookup failure")
        return types.SimpleNamespace(uuid=uuid)

    def nvmlDeviceGetIndex(self, handle):
        self._require_initialized()
        if hasattr(handle, "index"):
            return handle.index
        if hasattr(handle, "uuid") and handle.uuid in self.uuid_index_by_value:
            return self.uuid_index_by_value[handle.uuid]
        raise self.NVMLError("index lookup failure")

    def nvmlDeviceGetUtilizationRates(self, handle):
        self._require_initialized()
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


@pytest.mark.parametrize("gpu_util", [-1, 101])
def test_monitor_treats_out_of_range_gpu_utilization_as_unknown(monkeypatch, gpu_util):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyNVML(gpu_util=gpu_util)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == [0]


def test_monitor_reinitializes_after_external_nvml_shutdown(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyNVML(gpu_util=73)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) == 73
    dummy.nvmlShutdown()
    assert monitor.get_gpu_utilization(1) == 73
    assert dummy.init_calls == 2
    assert dummy.queried_indexes == [0, 1]


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


def test_monitor_returns_none_for_duplicate_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_equivalent_numeric_cuda_visible_devices(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,00")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_non_ascii_digit_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "\u00b2")
    dummy = DummyNVML(gpu_util=99, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_oversized_numeric_cuda_visible_devices(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", OVERSIZED_NUMERIC_TOKEN)
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_non_ascii_cuda_visible_devices_uuid_token(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-\u00b2")
    dummy = DummyNVML(gpu_util=99, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_duplicate_uuid_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123,GPU-abc123")
    dummy = DummyNVML(gpu_util=99, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_case_insensitive_duplicate_uuid_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123,gpu-abc123")
    dummy = DummyNVML(gpu_util=99, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_repeated_empty_tokens_without_handle_lookup(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,,")
    dummy = DummyNVML(util_by_index={0: 41})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_prior_token_is_invalid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1,2")
    dummy = DummyNVML(util_by_index={2: 99})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(2) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_later_token_is_empty_without_handle_lookup(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,,2")
    dummy = DummyNVML(util_by_index={2: 99})
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_later_numeric_index_is_invalid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,2")
    dummy = DummyNVML(util_by_index={0: 41, 2: 99}, count=8)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_when_later_uuid_token_is_unresolved(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-typo")
    dummy = DummyNVML(
        util_by_index={0: 41},
        support_uuid_lookup=True,
        fail_uuid_lookup=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-typo", b"GPU-typo"]


def test_monitor_returns_none_for_numeric_uuid_alias_to_same_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,GPU-zero")
    dummy = DummyNVML(
        support_uuid_lookup=True,
        uuid_index_by_value={"GPU-zero": 0},
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == [0]
    assert dummy.queried_uuids == ["GPU-zero"]


def test_monitor_returns_none_for_empty_cuda_visible_devices_token(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,,5")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(1) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == []


def test_monitor_returns_none_for_negative_mask_mixed_with_index(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1,0")
    dummy = DummyNVML(gpu_util=99)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
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


def test_monitor_retries_uuid_token_as_bytes_after_type_error(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(
        gpu_util=99,
        util_by_uuid={b"GPU-abc123": 61},
        support_uuid_lookup=True,
        uuid_requires_bytes=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) == 61
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123", b"GPU-abc123"]


def test_monitor_retries_uuid_token_as_bytes_after_nvml_error(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(
        gpu_util=99,
        util_by_uuid={b"GPU-abc123": 62},
        support_uuid_lookup=True,
        uuid_string_raises_nvml=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) == 62
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123", b"GPU-abc123"]


def test_monitor_returns_none_when_uuid_lookup_type_errors_for_all_inputs(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(
        support_uuid_lookup=True,
        uuid_always_type_error=True,
    )
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123", b"GPU-abc123"]


def test_monitor_returns_none_when_uuid_lookup_fails(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy = DummyNVML(fail_uuid_lookup=True, support_uuid_lookup=True)
    monitor = NVMLMonitor(dummy)

    assert monitor.get_gpu_utilization(0) is None
    assert dummy.queried_indexes == []
    assert dummy.queried_uuids == ["GPU-abc123", b"GPU-abc123"]


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
    assert dummy.queried_uuids == ["GPU-abc123", b"GPU-abc123"]
