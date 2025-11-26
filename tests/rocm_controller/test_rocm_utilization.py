import sys

import pytest

from keep_gpu.single_gpu_controller import rocm_gpu_controller as rgc


@pytest.mark.rocm
def test_query_rocm_utilization_with_mock(monkeypatch, rocm_available):
    if not rocm_available:
        pytest.skip("ROCm stack not available")

    class DummyRocmSMI:
        calls = 0

        @classmethod
        def rsmi_init(cls):
            cls.calls += 1

        @staticmethod
        def rsmi_dev_busy_percent_get(index):
            assert index == 1
            return 42

        @classmethod
        def rsmi_shut_down(cls):
            cls.calls += 1

    # Ensure the counter is reset to avoid leaking state between tests
    DummyRocmSMI.calls = 0
    monkeypatch.setitem(sys.modules, "rocm_smi", DummyRocmSMI)
    util = rgc._query_rocm_utilization(1)
    assert util == 42
    assert DummyRocmSMI.calls == 2  # init + shutdown
    # Reset after test for cleanliness
    DummyRocmSMI.calls = 0
