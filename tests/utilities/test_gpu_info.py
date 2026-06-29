from __future__ import annotations

import sys

import pytest

from keep_gpu.utilities import gpu_info


class DummyNVMLMemory:
    def __init__(self, total: int, used: int):
        self.total = total
        self.used = used


class DummyNVMLUtil:
    def __init__(self, gpu: int):
        self.gpu = gpu


class MultiGPUDummyNVML:
    def __init__(self, count: int, uuid_to_index: dict[object, int] | None = None):
        self.count = count
        self.uuid_to_index = uuid_to_index or {}
        self.queried_indexes = []
        self.queried_uuids = []
        self.shutdown_calls = 0

    @staticmethod
    def nvmlInit():
        return None

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        self.queried_indexes.append(index)
        return index

    def nvmlDeviceGetHandleByUUID(self, uuid):
        self.queried_uuids.append(uuid)
        return self.uuid_to_index[uuid]

    @staticmethod
    def nvmlDeviceGetIndex(handle):
        return handle

    @staticmethod
    def nvmlDeviceGetUUID(handle):
        return f"GPU-mock-{handle}".encode()

    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle):
        return DummyNVMLMemory(total=4096 + handle, used=1024 + handle)

    @staticmethod
    def nvmlDeviceGetUtilizationRates(handle):
        return DummyNVMLUtil(gpu=40 + handle)

    @staticmethod
    def nvmlDeviceGetName(handle):
        return f"Mock GPU {handle}".encode()

    def nvmlShutdown(self):
        self.shutdown_calls += 1


class DummyTorchCudaROCm:
    def __init__(
        self,
        count: int = 2,
        *,
        set_device_errors: "dict[int, Exception] | None" = None,
    ):
        self.count = count
        self.current = 0
        self.set_device_attempts: list[int] = []
        self.set_devices: list[int] = []
        self.set_device_errors = set_device_errors or {}

    @staticmethod
    def is_available():
        return True

    def current_device(self):
        return self.current

    def device_count(self):
        return self.count

    def set_device(self, idx):
        self.set_device_attempts.append(idx)
        if idx in self.set_device_errors:
            raise self.set_device_errors[idx]
        self.set_devices.append(idx)
        self.current = idx

    @staticmethod
    def mem_get_info():
        return (50, 100)

    @staticmethod
    def get_device_name(idx):
        return f"ROCm {idx}"


class DummyTorchCudaCUDA:
    def __init__(
        self,
        count: int = 1,
        *,
        available: bool = True,
        device_count_error: Exception | None = None,
        set_device_error: Exception | None = None,
    ):
        self.count = count
        self.available = available
        self.device_count_error = device_count_error
        self.set_device_error = set_device_error
        self.current = 0
        self.set_devices: list[int] = []

    def is_available(self):
        return self.available

    def current_device(self):
        return self.current

    def device_count(self):
        if self.device_count_error is not None:
            raise self.device_count_error
        return self.count

    def set_device(self, idx):
        if self.set_device_error is not None:
            raise self.set_device_error
        self.set_devices.append(idx)
        self.current = idx

    @staticmethod
    def mem_get_info():
        return (50, 100)

    @staticmethod
    def get_device_name(idx):
        return f"CUDA {idx}"


class DummyROCMSMI:
    def __init__(self, util_by_index: dict[int, int] | None = None, count: int = 8):
        self.util_by_index = util_by_index or {}
        self.count = count
        self.queried_indexes: list[int] = []
        self.init_calls = 0
        self.shutdown_calls = 0

    def rsmi_init(self):
        self.init_calls += 1

    def rsmi_num_monitor_devices(self):
        return self.count

    def rsmi_dev_busy_percent_get(self, idx):
        self.queried_indexes.append(idx)
        return self.util_by_index.get(idx, 40 + idx)

    def rsmi_shut_down(self):
        self.shutdown_calls += 1


def _install_rocm_gpu_info_mocks(
    monkeypatch,
    *,
    count: int = 2,
    util_by_index: dict[int, int] | None = None,
    smi_count: int = 8,
    set_device_errors: "dict[int, Exception] | None" = None,
):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    dummy_rocm = DummyROCMSMI(util_by_index=util_by_index, count=smi_count)
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy_rocm)

    dummy_cuda = DummyTorchCudaROCm(
        count=count,
        set_device_errors=set_device_errors,
    )
    dummy_torch = type(
        "T",
        (),
        {
            "cuda": dummy_cuda,
            "version": type("V", (), {"hip": "6.0"}),
            "backends": type(
                "Backends",
                (),
                {
                    "mps": type(
                        "MPSBackend", (), {"is_available": staticmethod(lambda: False)}
                    )
                },
            ),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)
    return dummy_rocm, dummy_cuda


def _install_cuda_gpu_info_mocks(
    monkeypatch,
    *,
    count: int = 1,
    available: bool = True,
    device_count_error: Exception | None = None,
    set_device_error: Exception | None = None,
    has_version: bool = True,
):
    monkeypatch.setitem(sys.modules, "rocm_smi", None)
    dummy_cuda = DummyTorchCudaCUDA(
        count=count,
        available=available,
        device_count_error=device_count_error,
        set_device_error=set_device_error,
    )
    torch_attrs = {
        "cuda": dummy_cuda,
        "backends": type(
            "Backends",
            (),
            {
                "mps": type(
                    "MPSBackend", (), {"is_available": staticmethod(lambda: False)}
                )
            },
        ),
    }
    if has_version:
        torch_attrs["version"] = type("V", (), {"hip": None})
    dummy_torch = type("T", (), torch_attrs)
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)
    return dummy_cuda


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
    _install_cuda_gpu_info_mocks(monkeypatch, count=1)

    infos = gpu_info.get_gpu_info()
    assert len(infos) == 1
    info = infos[0]
    assert info["name"] == "Mock GPU"
    assert info["memory_total"] == 2048
    assert info["memory_used"] == 1024
    assert info["utilization"] == 55
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_maps_cuda_visible_devices_to_visible_ordinals(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 0")
    dummy_nvml = MultiGPUDummyNVML(count=4)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=2)

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0, 1]
    assert [info["visible_id"] for info in infos] == [0, 1]
    assert [info["physical_id"] for info in infos] == [2, 0]
    assert [info["name"] for info in infos] == ["Mock GPU 2", "Mock GPU 0"]
    assert [info["utilization"] for info in infos] == [42, 40]
    assert dummy_nvml.queried_indexes == [2, 0]
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_maps_cuda_uuid_mask_to_visible_ordinal(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc123")
    dummy_nvml = MultiGPUDummyNVML(count=4, uuid_to_index={"GPU-abc123": 2})
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=1)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["id"] == 0
    assert infos[0]["visible_id"] == 0
    assert infos[0]["physical_id"] == 2
    assert infos[0]["uuid"] == "GPU-mock-2"
    assert infos[0]["name"] == "Mock GPU 2"
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.queried_uuids == ["GPU-abc123"]
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_empty_cuda_visible_devices_hides_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    dummy_nvml = MultiGPUDummyNVML(count=2)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_negative_cuda_visible_devices_hides_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    dummy_nvml = MultiGPUDummyNVML(count=2)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_unresolved_uuid_mask_does_not_list_placeholder(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-typo")
    dummy_nvml = MultiGPUDummyNVML(count=2)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.queried_uuids == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_malformed_cuda_visible_devices_hides_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,,2")
    dummy_nvml = MultiGPUDummyNVML(count=3)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


@pytest.mark.parametrize(
    "mask",
    ["", "-1", "0,,2", "0,0", "0,00", "GPU-abc123,gpu-abc123", "-1,0", "\u00b2"],
)
def test_get_gpu_info_cuda_hidden_or_invalid_mask_does_not_fall_back_to_torch(
    monkeypatch, mask
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    dummy_cuda = _install_cuda_gpu_info_mocks(monkeypatch, count=1)

    assert gpu_info.get_gpu_info() == []
    assert dummy_cuda.set_devices == []


def test_get_gpu_info_nvml_out_of_range_cuda_visible_devices_hides_all(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,99,2")
    dummy_nvml = MultiGPUDummyNVML(count=3)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_duplicate_cuda_visible_devices_hides_all(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,0")
    dummy_nvml = MultiGPUDummyNVML(count=2)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_nvml_duplicate_uuid_mask_hides_all_when_torch_has_no_devices(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    dummy_nvml = MultiGPUDummyNVML(
        count=3,
        uuid_to_index={"GPU-a": 2, "GPU-b": 2},
    )
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.queried_uuids == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_rocm_maps_visible_ids_to_physical_smi_indexes(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,4")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_rocm, _ = _install_rocm_gpu_info_mocks(
        monkeypatch,
        util_by_index={2: 82, 4: 84},
    )

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0, 1]
    assert [info["visible_id"] for info in infos] == [0, 1]
    assert [info["physical_id"] for info in infos] == [2, 4]
    assert [info["name"] for info in infos] == ["ROCm 0", "ROCm 1"]
    assert [info["utilization"] for info in infos] == [82, 84]
    assert dummy_rocm.queried_indexes == [2, 4]
    assert dummy_rocm.shutdown_calls == 1


def test_get_gpu_info_rocm_composes_rocr_base_and_hip_visible_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,4")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_rocm, _ = _install_rocm_gpu_info_mocks(
        monkeypatch,
        count=1,
        util_by_index={4: 94},
    )

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0]
    assert [info["visible_id"] for info in infos] == [0]
    assert [info["physical_id"] for info in infos] == [4]
    assert [info["utilization"] for info in infos] == [94]
    assert dummy_rocm.queried_indexes == [4]
    assert dummy_rocm.shutdown_calls == 1


def test_get_gpu_info_rocm_hides_unstartable_visible_ordinals(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_rocm, dummy_cuda = _install_rocm_gpu_info_mocks(
        monkeypatch,
        count=2,
        util_by_index={0: 70, 1: 91},
        set_device_errors={1: RuntimeError("cannot select rocm:1")},
    )

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0]
    assert [info["visible_id"] for info in infos] == [0]
    assert [info["physical_id"] for info in infos] == [0]
    assert [info["name"] for info in infos] == ["ROCm 0"]
    assert [info["utilization"] for info in infos] == [70]
    assert dummy_rocm.queried_indexes == [0]
    assert dummy_rocm.shutdown_calls == 1
    assert dummy_cuda.set_device_attempts == [0, 1, 0]
    assert dummy_cuda.set_devices == [0, 0]


def test_get_gpu_info_rocm_unresolved_mask_keeps_visible_records_without_utilization(
    monkeypatch,
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,,2")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_rocm, _ = _install_rocm_gpu_info_mocks(monkeypatch)

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0, 1]
    assert [info["visible_id"] for info in infos] == [0, 1]
    assert all("physical_id" not in info for info in infos)
    assert [info["utilization"] for info in infos] == [None, None]
    assert dummy_rocm.queried_indexes == []
    assert dummy_rocm.shutdown_calls == 1


def test_get_gpu_info_prefers_rocm_when_hip_torch_and_nvml_are_both_available(
    monkeypatch,
):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_nvml = MultiGPUDummyNVML(count=2)
    dummy_rocm, _ = _install_rocm_gpu_info_mocks(
        monkeypatch,
        count=1,
        util_by_index={0: 71},
    )
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "rocm"
    assert infos[0]["visible_id"] == 0
    assert infos[0]["physical_id"] == 0
    assert infos[0]["name"] == "ROCm 0"
    assert infos[0]["utilization"] == 71
    assert dummy_rocm.queried_indexes == [0]
    assert dummy_rocm.shutdown_calls == 1
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 0


def test_get_gpu_info_hip_torch_skips_nvml_when_rocm_smi_unavailable(monkeypatch):
    dummy_nvml = MultiGPUDummyNVML(count=2)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)
    dummy_cuda = DummyTorchCudaROCm(count=1)
    dummy_torch = type(
        "T",
        (),
        {
            "cuda": dummy_cuda,
            "version": type("V", (), {"hip": "6.0"}),
            "backends": type(
                "Backends",
                (),
                {
                    "mps": type(
                        "MPSBackend", (), {"is_available": staticmethod(lambda: False)}
                    )
                },
            ),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "rocm"
    assert infos[0]["visible_id"] == 0
    assert infos[0]["utilization"] is None
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 0


def test_get_gpu_info_cuda_nvml_allows_torch_without_version_attribute_when_startable(
    monkeypatch,
):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=1, has_version=False)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "cuda"
    assert dummy_nvml.queried_indexes == [0]
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_cuda_nvml_hides_devices_when_torch_unavailable(monkeypatch):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, available=False)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_cuda_nvml_hides_devices_when_torch_count_is_zero(monkeypatch):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(monkeypatch, count=0)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_cuda_nvml_hides_devices_when_torch_count_fails(monkeypatch):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    _install_cuda_gpu_info_mocks(
        monkeypatch,
        device_count_error=RuntimeError("cuda runtime unavailable"),
    )

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1


def test_get_gpu_info_cuda_nvml_hides_devices_when_torch_set_device_fails(
    monkeypatch,
):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    dummy_cuda = _install_cuda_gpu_info_mocks(
        monkeypatch,
        count=1,
        set_device_error=RuntimeError("cannot select cuda:0"),
    )

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1
    assert dummy_cuda.set_devices == []


def test_get_gpu_info_cuda_nvml_mask_mismatch_falls_back_to_torch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    dummy_nvml = MultiGPUDummyNVML(count=3)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    dummy_cuda = _install_cuda_gpu_info_mocks(monkeypatch, count=1)

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0]
    assert [info["visible_id"] for info in infos] == [0]
    assert [info["platform"] for info in infos] == ["cuda"]
    assert "physical_id" not in infos[0]
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1
    assert dummy_cuda.set_devices == [0, 0]


def test_get_gpu_info_cuda_nvml_mismatch_prefers_torch_over_rocm_smi(
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2")
    dummy_nvml = MultiGPUDummyNVML(count=3)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    dummy_rocm = DummyROCMSMI(util_by_index={0: 77})
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy_rocm)
    dummy_cuda = _install_cuda_gpu_info_mocks(monkeypatch, count=1)
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy_rocm)

    infos = gpu_info.get_gpu_info()

    assert [info["id"] for info in infos] == [0]
    assert [info["visible_id"] for info in infos] == [0]
    assert [info["platform"] for info in infos] == ["cuda"]
    assert "physical_id" not in infos[0]
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1
    assert dummy_rocm.init_calls == 0
    assert dummy_rocm.queried_indexes == []
    assert dummy_cuda.set_devices == [0, 0]


def test_get_gpu_info_cuda_unstartable_torch_does_not_fall_back_to_rocm_smi(
    monkeypatch,
):
    dummy_nvml = MultiGPUDummyNVML(count=1)
    monkeypatch.setitem(sys.modules, "pynvml", dummy_nvml)
    dummy_rocm = DummyROCMSMI(util_by_index={0: 77})
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy_rocm)
    dummy_cuda = _install_cuda_gpu_info_mocks(
        monkeypatch,
        count=1,
        set_device_error=RuntimeError("cannot select cuda:0"),
    )
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy_rocm)

    assert gpu_info.get_gpu_info() == []
    assert dummy_nvml.queried_indexes == []
    assert dummy_nvml.shutdown_calls == 1
    assert dummy_rocm.init_calls == 0
    assert dummy_rocm.queried_indexes == []
    assert dummy_cuda.set_devices == []


def test_get_gpu_info_torch_fallback_allows_missing_version_attribute(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)
    dummy_cuda = DummyTorchCudaROCm(count=1)
    dummy_torch = type(
        "T",
        (),
        {
            "cuda": dummy_cuda,
            "backends": type(
                "Backends",
                (),
                {
                    "mps": type(
                        "MPSBackend", (), {"is_available": staticmethod(lambda: False)}
                    )
                },
            ),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "cuda"
    assert infos[0]["visible_id"] == 0
    assert dummy_cuda.set_devices == [0, 0]


@pytest.mark.rocm
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


def test_get_gpu_info_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)

    class DummyCuda:
        @staticmethod
        def is_available():
            return False

    class DummyMPSBackend:
        @staticmethod
        def is_available():
            return True

    class DummyMPS:
        @staticmethod
        def current_allocated_memory():
            return 128

        @staticmethod
        def driver_allocated_memory():
            return 256

        @staticmethod
        def recommended_max_memory():
            return 1024

    dummy_torch = type(
        "T",
        (),
        {
            "cuda": DummyCuda,
            "backends": type("Backends", (), {"mps": DummyMPSBackend}),
            "mps": DummyMPS,
            "version": type("V", (), {"hip": None}),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    infos = gpu_info.get_gpu_info()

    assert infos == [
        {
            "id": 0,
            "visible_id": 0,
            "platform": "macm",
            "name": "Apple Silicon GPU",
            "memory_total": 1024,
            "memory_used": 256,
            "utilization": None,
            "memory_allocated": 128,
        }
    ]


def test_get_gpu_info_mps_allows_missing_memory_methods(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)

    class DummyCuda:
        @staticmethod
        def is_available():
            return False

    class DummyMPSBackend:
        @staticmethod
        def is_available():
            return True

    dummy_torch = type(
        "T",
        (),
        {
            "cuda": DummyCuda,
            "backends": type("Backends", (), {"mps": DummyMPSBackend}),
            "version": type("V", (), {"hip": None}),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "macm"
    assert infos[0]["visible_id"] == 0
    assert infos[0]["memory_total"] is None
    assert infos[0]["memory_used"] is None


def test_get_gpu_info_mps_fallback_survives_torch_cuda_probe_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "rocm_smi", None)

    class BrokenCuda:
        @staticmethod
        def is_available():
            raise RuntimeError("cuda probe failed")

    class DummyMPSBackend:
        @staticmethod
        def is_available():
            return True

    class DummyMPS:
        @staticmethod
        def current_allocated_memory():
            return 64

        @staticmethod
        def driver_allocated_memory():
            return 128

        @staticmethod
        def recommended_max_memory():
            return 512

    dummy_torch = type(
        "T",
        (),
        {
            "cuda": BrokenCuda,
            "backends": type("Backends", (), {"mps": DummyMPSBackend}),
            "mps": DummyMPS,
            "version": type("V", (), {"hip": None}),
        },
    )
    monkeypatch.setattr(gpu_info, "torch", dummy_torch)

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "macm"
    assert infos[0]["visible_id"] == 0
    assert infos[0]["memory_total"] == 512
    assert infos[0]["memory_used"] == 128
