from __future__ import annotations

import sys

import pytest

from keep_gpu.single_gpu_controller import rocm_gpu_controller as rocm_module
from keep_gpu.single_gpu_controller.rocm_gpu_controller import RocmGPUController


class DummyRocmSMI:
    def __init__(self, util_by_index: dict[int, int] | None = None, count: int = 8):
        self.util_by_index = util_by_index or {}
        self.count = count
        self.queried_indexes: list[int] = []

    def rsmi_num_monitor_devices(self):
        return self.count

    def rsmi_dev_busy_percent_get(self, index):
        self.queried_indexes.append(index)
        return self.util_by_index.get(index, 40 + index)


def _controller_with_dummy_smi(monkeypatch, rank: int, dummy: DummyRocmSMI):
    monkeypatch.setitem(sys.modules, "rocm_smi", dummy)
    monkeypatch.setattr(
        rocm_module.torch.cuda,
        "device_count",
        lambda: max(rank + 1, 1),
    )
    return RocmGPUController(rank=rank, vram_to_keep=4)


def test_rocm_utilization_defaults_visible_rank_to_smi_index(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(util_by_index={1: 71})
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() == 71
    assert dummy.queried_indexes == [1]


def test_rocm_utilization_maps_hip_visible_rank_to_smi_index(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,5")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(util_by_index={5: 88})
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() == 88
    assert dummy.queried_indexes == [5]


def test_rocm_utilization_composes_rocr_base_and_hip_visible_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,4")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(util_by_index={4: 91})
    controller = _controller_with_dummy_smi(monkeypatch, rank=0, dummy=dummy)

    assert controller._query_utilization() == 91
    assert dummy.queried_indexes == [4]


def test_rocm_utilization_uses_cuda_mask_when_hip_is_unset(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(util_by_index={5: 89})
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() == 89
    assert dummy.queried_indexes == [5]


def test_rocm_utilization_treats_conflicting_hip_cuda_masks_as_unavailable(
    monkeypatch,
):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,6")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI()
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []


def test_rocm_utilization_treats_malformed_mask_as_unavailable(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,,2")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI()
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []


@pytest.mark.parametrize(
    "mask_name",
    ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
)
def test_rocm_utilization_treats_non_ascii_digit_mask_as_unavailable(
    monkeypatch, mask_name
):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv(mask_name, "\u00b2")
    dummy = DummyRocmSMI()
    controller = _controller_with_dummy_smi(monkeypatch, rank=0, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []


def test_rocm_utilization_treats_out_of_range_mask_as_unavailable(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,9")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(count=4)
    controller = _controller_with_dummy_smi(monkeypatch, rank=1, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []


def test_rocm_utilization_treats_unresolved_uuid_mask_as_unavailable(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-abc123")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI()
    controller = _controller_with_dummy_smi(monkeypatch, rank=0, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []
