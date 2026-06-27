import os

import pytest
import typer

from keep_gpu import cli


def test_parse_gpu_ids_rejects_negative_values():
    with pytest.raises(typer.BadParameter, match="non-negative integers"):
        cli._parse_gpu_ids("0,-1")


def test_parse_gpu_ids_rejects_duplicate_values():
    with pytest.raises(typer.BadParameter, match="duplicate values"):
        cli._parse_gpu_ids("0,1,0")


def test_apply_legacy_threshold_none():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", None, -1)
    assert vram == "1GiB"
    assert threshold == -1
    assert mode is None


def test_apply_legacy_threshold_numeric():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "25", -1)
    assert vram == "1GiB"
    assert threshold == 25
    assert mode == "busy"


def test_apply_legacy_threshold_memory_string():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "2GiB", -1)
    assert vram == "2GiB"
    assert threshold == -1
    assert mode == "vram"


def test_validate_cli_busy_threshold_rejects_legacy_value_above_percent_range():
    vram, threshold, mode = cli._apply_legacy_threshold("1GiB", "101", -1)

    assert vram == "1GiB"
    assert threshold == 101
    assert mode == "busy"
    with pytest.raises(
        typer.BadParameter,
        match="busy_threshold must be -1 or an integer between 0 and 100",
    ):
        cli._validate_cli_busy_threshold(threshold)


def test_run_blocking_preserves_cuda_visible_devices_for_gpu_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    captured = {}

    class DummyGlobalController:
        def __init__(self, *, gpu_ids, interval, vram_to_keep, busy_threshold):
            captured["gpu_ids"] = gpu_ids
            captured["interval"] = interval
            captured["vram_to_keep"] = vram_to_keep
            captured["busy_threshold"] = busy_threshold

        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["exited"] = True

    import keep_gpu.global_gpu_controller.global_gpu_controller as global_module

    monkeypatch.setattr(global_module, "GlobalGPUController", DummyGlobalController)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)

    cli._run_blocking(
        interval=1,
        gpu_ids="3",
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=-1,
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    assert captured == {
        "gpu_ids": [3],
        "interval": 1,
        "vram_to_keep": "1MiB",
        "busy_threshold": -1,
        "entered": True,
        "exited": True,
    }
