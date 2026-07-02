import logging
import os

import pytest
import typer
from typer.testing import CliRunner

from keep_gpu import cli

runner = CliRunner()


def test_parse_gpu_ids_treats_omitted_as_all_visible():
    assert cli._parse_gpu_ids(None) is None


@pytest.mark.parametrize("gpu_ids", ["", "   "])
def test_parse_gpu_ids_rejects_empty_values(gpu_ids):
    with pytest.raises(typer.BadParameter, match="gpu_ids must not be empty"):
        cli._parse_gpu_ids(gpu_ids)


def test_parse_gpu_ids_rejects_negative_values():
    with pytest.raises(typer.BadParameter, match="non-negative integers"):
        cli._parse_gpu_ids("0,-1")


def test_parse_gpu_ids_rejects_duplicate_values():
    with pytest.raises(typer.BadParameter, match="duplicate values"):
        cli._parse_gpu_ids("0,1,0")


@pytest.mark.parametrize("gpu_ids", ["1_000", "１２３", "+0", "-0", "-00"])
def test_parse_gpu_ids_rejects_non_canonical_numeric_tokens(gpu_ids):
    with pytest.raises(typer.BadParameter, match="Invalid characters in --gpu-ids"):
        cli._parse_gpu_ids(gpu_ids)


@pytest.mark.parametrize("interval", ["1_000", "１２３", "+1"])
def test_validate_cli_interval_rejects_non_canonical_numeric_tokens(interval):
    with pytest.raises(
        typer.BadParameter, match="interval must be finite and positive"
    ):
        cli._validate_cli_interval(interval)


def test_validate_cli_interval_preserves_exponent_plus_sign():
    assert cli._validate_cli_interval("1e+3") == 1000


@pytest.mark.parametrize("busy_threshold", ["1_0", "１２", "+25", "-0"])
def test_validate_cli_busy_threshold_rejects_non_canonical_numeric_tokens(
    busy_threshold,
):
    with pytest.raises(
        typer.BadParameter,
        match="busy_threshold must be -1 or an integer between 0 and 100",
    ):
        cli._validate_cli_busy_threshold(busy_threshold)


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


@pytest.mark.parametrize("legacy_threshold", ["1_0", "１２", "+25", "-0"])
def test_apply_legacy_threshold_rejects_non_canonical_numeric_tokens(
    legacy_threshold,
):
    with pytest.raises(
        typer.BadParameter,
        match="threshold must be an integer utilization value or a VRAM size",
    ):
        cli._apply_legacy_threshold("1GiB", legacy_threshold, -1)


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


def test_blocking_command_rejects_empty_gpu_ids_before_run_blocking(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run_blocking",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocking mode should not start")
        ),
    )

    result = runner.invoke(cli.app, ["--gpu-ids", ""])

    assert result.exit_code == 1
    assert "gpu_ids must not be empty" in result.output


def test_blocking_command_accepts_fractional_interval(monkeypatch):
    captured = {}

    def fake_run_blocking(interval, gpu_ids, vram, legacy_threshold, busy_threshold):
        captured["interval"] = interval
        captured["gpu_ids"] = gpu_ids
        captured["vram"] = vram
        captured["legacy_threshold"] = legacy_threshold
        captured["busy_threshold"] = busy_threshold

    monkeypatch.setattr(cli, "_run_blocking", fake_run_blocking)

    result = runner.invoke(cli.app, ["--interval", "0.5"])

    assert result.exit_code == 0
    assert captured["interval"] == 0.5


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


def test_run_blocking_defers_omitted_gpu_enumeration_to_global_controller(
    monkeypatch,
):
    import torch

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

    def fail_device_count():
        raise AssertionError("blocking mode should defer GPU enumeration")

    monkeypatch.setattr(torch.cuda, "device_count", fail_device_count)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)

    cli._run_blocking(
        interval=1,
        gpu_ids=None,
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=-1,
    )

    assert captured == {
        "gpu_ids": None,
        "interval": 1,
        "vram_to_keep": "1MiB",
        "busy_threshold": -1,
        "entered": True,
        "exited": True,
    }


@pytest.mark.parametrize(
    ("busy_threshold", "expected_message", "forbidden_message"),
    [
        (
            -1,
            "Busy threshold: unconditional (utilization backoff disabled)",
            "Busy threshold: -1%",
        ),
        (25, "Busy threshold: 25%", "unconditional"),
    ],
)
def test_run_blocking_logs_busy_threshold_semantically(
    monkeypatch, caplog, busy_threshold, expected_message, forbidden_message
):
    class DummyGlobalController:
        def __init__(self, *, gpu_ids, interval, vram_to_keep, busy_threshold):
            self.gpu_ids = gpu_ids
            self.interval = interval
            self.vram_to_keep = vram_to_keep
            self.busy_threshold = busy_threshold

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    import keep_gpu.global_gpu_controller.global_gpu_controller as global_module

    monkeypatch.setattr(global_module, "GlobalGPUController", DummyGlobalController)

    def interrupt_sleep(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", interrupt_sleep)
    monkeypatch.setattr(cli.logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=cli.logger.name)

    cli._run_blocking(
        interval=1,
        gpu_ids="0",
        vram="1MiB",
        legacy_threshold=None,
        busy_threshold=busy_threshold,
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == cli.logger.name
    ]
    assert any(expected_message in message for message in messages)
    assert not any(forbidden_message in message for message in messages)
