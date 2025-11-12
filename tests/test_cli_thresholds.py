from keep_gpu import cli


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
