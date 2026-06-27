import pytest

from keep_gpu.utilities.session_config import (
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
)


def test_validate_interval_accepts_fractional_positive_seconds():
    assert validate_interval(0.05) == 0.05


def test_validate_interval_rejects_non_positive_values():
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(0)
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(-0.1)


def test_validate_gpu_ids_rejects_empty_list():
    with pytest.raises(ValueError, match="gpu_ids must select at least one GPU"):
        validate_gpu_ids([])


def test_validate_gpu_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="gpu_ids must not contain duplicate values"):
        validate_gpu_ids([0, 1, 0])


def test_validate_busy_threshold_only_allows_minus_one_as_negative():
    assert validate_busy_threshold(-1) == -1
    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        validate_busy_threshold(-2)


def test_validate_busy_threshold_accepts_percent_upper_bound():
    assert validate_busy_threshold(100) == 100


def test_validate_busy_threshold_rejects_values_above_percent_range():
    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        validate_busy_threshold(101)


@pytest.mark.parametrize("value", [True, False, 0.5, "25"])
def test_validate_busy_threshold_rejects_non_plain_integers(value):
    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        validate_busy_threshold(value)
