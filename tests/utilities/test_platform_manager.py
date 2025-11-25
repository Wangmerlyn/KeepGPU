import sys

from keep_gpu.utilities import platform_manager as pm


def _reset_cache(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", None)
    # isolate checks for each test
    monkeypatch.setattr(
        pm,
        "_PLATFORM_CHECKS",
        [
            (pm.ComputingPlatform.CUDA, lambda: False),
            (pm.ComputingPlatform.ROCM, lambda: False),
            (pm.ComputingPlatform.CPU, lambda: True),
        ],
    )


def test_env_override_cpu(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setenv("KEEP_GPU_PLATFORM", "cpu")
    assert pm.get_platform() == pm.ComputingPlatform.CPU


def test_invalid_override_falls_back(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setenv("KEEP_GPU_PLATFORM", "invalid")
    calls = {"count": 0}

    def fake_cuda():
        calls["count"] += 1
        return False

    monkeypatch.setattr(
        pm,
        "_PLATFORM_CHECKS",
        [
            (pm.ComputingPlatform.CUDA, fake_cuda),
            (pm.ComputingPlatform.CPU, lambda: True),
        ],
    )
    assert pm.get_platform() == pm.ComputingPlatform.CPU
    assert calls["count"] == 1


def test_cached_result_reused(monkeypatch):
    _reset_cache(monkeypatch)
    calls = {"count": 0}

    def _fake_check():
        calls["count"] += 1
        return True

    monkeypatch.setattr(
        pm,
        "_PLATFORM_CHECKS",
        [(pm.ComputingPlatform.CPU, _fake_check)],
    )

    assert pm.get_platform() == pm.ComputingPlatform.CPU
    assert pm.get_platform() == pm.ComputingPlatform.CPU
    assert calls["count"] == 1


def test_cuda_prefers_torch_over_hip(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": None}))
    assert pm._check_cuda() is True


def test_rocm_detects_hip(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": "6.0"}))
    assert pm._check_rocm() is True


def test_cuda_detection_falls_back_to_nvml(monkeypatch):
    _reset_cache(monkeypatch)
    # Force torch to look like ROCm build to ensure NVML takes precedence
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": "6.0"}))

    class DummyNVML:
        @staticmethod
        def nvmlInit():
            return None

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)
    assert pm._check_cuda() is True
