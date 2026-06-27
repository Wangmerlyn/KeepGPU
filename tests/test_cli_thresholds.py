import pytest

from keep_gpu import cli


def test_parse_gpu_ids_rejects_negative_values():
    with pytest.raises(Exception, match="non-negative integers"):
        cli._parse_gpu_ids("0,-1")


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
        Exception, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        cli._validate_cli_busy_threshold(threshold)
