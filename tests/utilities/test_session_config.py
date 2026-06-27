import pytest

from keep_gpu.utilities.session_config import (
    validate_busy_threshold,
    validate_interval,
)


def test_validate_interval_accepts_fractional_positive_seconds():
    assert validate_interval(0.05) == 0.05


def test_validate_interval_rejects_non_positive_values():
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(0)
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(-0.1)


def test_validate_busy_threshold_only_allows_minus_one_as_negative():
    assert validate_busy_threshold(-1) == -1
    with pytest.raises(ValueError, match="busy_threshold must be an integer >= -1"):
        validate_busy_threshold(-2)
