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


def test_cuda_detected_via_torch_non_hip_build(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": None}))
    assert pm._check_cuda() is True


def test_rocm_detects_hip(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": "6.0"}))
    assert pm._check_rocm() is True


def test_get_platform_prefers_rocm_torch_over_nvml(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", None)
    monkeypatch.delenv("KEEP_GPU_PLATFORM", raising=False)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": "6.0"}))

    class DummyNVML:
        @staticmethod
        def nvmlInit():
            return None

        @staticmethod
        def nvmlShutdown():
            return None

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)

    assert pm.get_platform() == pm.ComputingPlatform.ROCM


def test_cuda_detection_skips_nvml_for_rocm_torch_build(monkeypatch):
    _reset_cache(monkeypatch)

    def fail_cuda_probe():
        raise AssertionError("CUDA availability should not be probed for HIP builds")

    monkeypatch.setattr(pm.torch.cuda, "is_available", fail_cuda_probe)
    monkeypatch.setattr(pm.torch, "version", type("v", (), {"hip": "6.0"}))

    class DummyNVML:
        init_calls = 0

        @staticmethod
        def nvmlInit():
            DummyNVML.init_calls += 1
            return None

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)
    assert pm._check_cuda() is False
    assert DummyNVML.init_calls == 0


def test_cuda_detection_uses_torch_when_hip_attr_missing(monkeypatch):
    _reset_cache(monkeypatch)
    cuda_calls = 0

    def cuda_available():
        nonlocal cuda_calls
        cuda_calls += 1
        return True

    monkeypatch.setattr(pm.torch.cuda, "is_available", cuda_available)
    monkeypatch.delattr(pm.torch.version, "hip", raising=False)

    class DummyNVML:
        init_calls = 0

        @staticmethod
        def nvmlInit():
            DummyNVML.init_calls += 1
            raise RuntimeError("NVML should not be needed when torch sees CUDA")

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)

    assert pm._check_cuda() is True
    assert cuda_calls == 1
    assert DummyNVML.init_calls == 0


def test_cuda_detection_shuts_down_nvml_probe(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: False)

    class DummyNVML:
        init_calls = 0
        shutdown_calls = 0

        @classmethod
        def nvmlInit(cls):
            cls.init_calls += 1

        @classmethod
        def nvmlShutdown(cls):
            cls.shutdown_calls += 1

    monkeypatch.setitem(sys.modules, "pynvml", DummyNVML)

    assert pm._check_cuda() is True
    assert DummyNVML.init_calls == 1
    assert DummyNVML.shutdown_calls == 1


def test_rocm_detection_uses_rsmi_api(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setattr(pm.torch.cuda, "is_available", lambda: False)

    class DummyROCM:
        init_calls = 0
        shutdown_calls = 0

        @classmethod
        def rsmi_init(cls):
            cls.init_calls += 1

        @classmethod
        def rsmi_shut_down(cls):
            cls.shutdown_calls += 1

    monkeypatch.setitem(sys.modules, "rocm_smi", DummyROCM)

    assert pm._check_rocm() is True
    assert DummyROCM.init_calls == 1
    assert DummyROCM.shutdown_calls == 1
