from keep_gpu.utilities import platform_manager as pm


def _reset_cache(monkeypatch):
    monkeypatch.setattr(pm, "_cached_platform", None)


def test_env_override_cpu(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setenv("KEEP_GPU_PLATFORM", "cpu")
    # Ensure no real detection runs
    monkeypatch.setattr(
        pm,
        "_PLATFORM_CHECKS",
        [(pm.ComputingPlatform.CPU, lambda: True)],
    )
    assert pm.get_platform() == pm.ComputingPlatform.CPU


def test_invalid_override_falls_back(monkeypatch):
    _reset_cache(monkeypatch)
    monkeypatch.setenv("KEEP_GPU_PLATFORM", "invalid")
    monkeypatch.setattr(
        pm,
        "_PLATFORM_CHECKS",
        [
            (pm.ComputingPlatform.CUDA, lambda: False),
            (pm.ComputingPlatform.CPU, lambda: True),
        ],
    )
    assert pm.get_platform() == pm.ComputingPlatform.CPU


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
