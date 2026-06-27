from keep_gpu.utilities.humanized_input import parse_size, parse_vram_to_elements


def test_parse_size_digit_only_string_means_bytes():
    assert parse_size("1073741824") == 268_435_456


def test_parse_size_human_units_still_mean_bytes():
    assert parse_size("1GiB") == 268_435_456
    assert parse_size("512MB") == 128_000_000


def test_parse_vram_integer_means_bytes():
    assert parse_vram_to_elements(1_073_741_824) == 268_435_456
