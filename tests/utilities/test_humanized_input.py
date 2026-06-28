import pytest

from keep_gpu.utilities.humanized_input import (
    PUBLIC_VRAM_MAX_BYTES,
    parse_size,
    parse_vram_to_elements,
)


def test_parse_size_digit_only_string_means_bytes():
    assert parse_size("1073741824") == 268_435_456


def test_parse_size_human_units_still_mean_bytes():
    assert parse_size("1GiB") == 268_435_456
    assert parse_size("512MB") == 128_000_000


def test_parse_vram_integer_means_bytes():
    assert parse_vram_to_elements(1_073_741_824) == 268_435_456


def test_parse_vram_accepts_public_maximum_bytes():
    assert parse_vram_to_elements(PUBLIC_VRAM_MAX_BYTES) == PUBLIC_VRAM_MAX_BYTES // 4
    assert parse_size(str(PUBLIC_VRAM_MAX_BYTES)) == PUBLIC_VRAM_MAX_BYTES // 4


def test_parse_vram_rejects_values_above_public_maximum():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_vram_to_elements(PUBLIC_VRAM_MAX_BYTES + 1)
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size(str(PUBLIC_VRAM_MAX_BYTES + 1))


def test_parse_size_rejects_decimal_string_above_public_maximum():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size(f"{PUBLIC_VRAM_MAX_BYTES}.1")


def test_parse_size_rejects_unit_decimal_string_above_public_maximum():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size("1048576.0000000000000000000000001GiB")


def test_parse_vram_rejects_oversized_integer_without_overflow():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_vram_to_elements(10**1000)


def test_parse_size_rejects_oversized_digit_only_string_without_overflow():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size("9" * 500)


def test_parse_size_rejects_digit_string_above_python_int_limit_cleanly():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size("9" * 4301)


def test_parse_size_rejects_oversized_human_unit_string_without_overflow():
    with pytest.raises(ValueError, match="vram must be no more than"):
        parse_size(("9" * 500) + "GiB")
