import math
import uuid

import pytest

from keep_gpu.utilities import session_config
from keep_gpu.utilities.session_config import (
    PUBLIC_INTERVAL_MAX_SECONDS,
    is_memory_byte_or_none,
    is_memory_byte_pair_or_none,
    normalize_memory_byte_pair,
    normalize_memory_bytes,
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
    validate_rank_type,
    validate_visible_rank,
)


def test_validate_interval_accepts_fractional_positive_seconds():
    assert validate_interval(0.05) == 0.05


def test_validate_interval_rejects_non_positive_values():
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(0)
    with pytest.raises(ValueError, match="interval must be positive"):
        validate_interval(-0.1)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_validate_interval_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="interval must be finite and positive"):
        validate_interval(value)


def test_validate_interval_accepts_public_maximum_seconds():
    assert validate_interval(PUBLIC_INTERVAL_MAX_SECONDS) == PUBLIC_INTERVAL_MAX_SECONDS


def test_validate_interval_rejects_above_public_maximum_seconds():
    with pytest.raises(ValueError, match="interval must be no more than"):
        validate_interval(PUBLIC_INTERVAL_MAX_SECONDS + 1)


def test_validate_interval_rejects_oversized_integer_without_overflow():
    with pytest.raises(ValueError, match="interval must be no more than"):
        validate_interval(10**1000)


def test_validate_gpu_ids_rejects_empty_list():
    with pytest.raises(ValueError, match="gpu_ids must select at least one GPU"):
        validate_gpu_ids([])


def test_validate_gpu_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="gpu_ids must not contain duplicate values"):
        validate_gpu_ids([0, 1, 0])


def test_validate_visible_rank_accepts_visible_ordinal():
    assert validate_visible_rank(1, 2) == 1


def test_validate_rank_type_accepts_plain_integer_rank():
    assert validate_rank_type(0) == 0


@pytest.mark.parametrize("rank", [True, 1.5, "1"])
def test_validate_rank_type_rejects_non_plain_integer_rank(rank):
    with pytest.raises(TypeError, match="rank must be an integer"):
        validate_rank_type(rank)


@pytest.mark.parametrize("rank", [True, 1.5, "1"])
def test_validate_visible_rank_rejects_non_plain_integer_rank(rank):
    with pytest.raises(TypeError, match="rank must be an integer"):
        validate_visible_rank(rank, 2)


@pytest.mark.parametrize("visible_count", [True, -1, 1.5, "2"])
def test_validate_visible_rank_rejects_invalid_visible_count(visible_count):
    with pytest.raises(
        ValueError, match="visible device count must be a non-negative integer"
    ):
        validate_visible_rank(0, visible_count)


def test_validate_visible_rank_rejects_zero_visible_count_with_clear_message():
    with pytest.raises(ValueError, match="no visible GPUs are available"):
        validate_visible_rank(0, 0)


def test_validate_visible_rank_selection_failures_are_typed_value_errors():
    error_type = getattr(session_config, "VisibleRankValidationError", None)
    assert error_type is not None
    assert issubclass(error_type, ValueError)

    with pytest.raises(error_type, match="no visible GPUs are available"):
        validate_visible_rank(0, 0)
    with pytest.raises(error_type, match="rank must be a visible device ordinal"):
        validate_visible_rank(2, 2)


@pytest.mark.parametrize("rank", [-1, 2])
def test_validate_visible_rank_rejects_out_of_range_rank(rank):
    with pytest.raises(ValueError, match="rank must be a visible device ordinal"):
        validate_visible_rank(rank, 2)


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


@pytest.mark.parametrize("value", [None, 0, 1, 1024])
def test_memory_byte_fields_accept_null_or_non_negative_plain_ints(value):
    assert is_memory_byte_or_none(value) is True
    expected = value if isinstance(value, int) else None
    assert normalize_memory_bytes(value) == expected


@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1"])
def test_memory_byte_fields_reject_non_plain_or_negative_values(value):
    assert is_memory_byte_or_none(value) is False
    assert normalize_memory_bytes(value) is None


@pytest.mark.parametrize(
    ("total", "used", "expected_total", "expected_used", "valid"),
    [
        (1024, 512, 1024, 512, True),
        (1024, None, 1024, None, True),
        (None, 512, None, 512, True),
        (None, None, None, None, True),
        (1024, 2048, 1024, None, False),
        (-1, 0, None, 0, False),
        (1024, -1, 1024, None, False),
    ],
)
def test_memory_byte_pair_rejects_impossible_used_over_total(
    total, used, expected_total, expected_used, valid
):
    assert normalize_memory_byte_pair(total, used) == (expected_total, expected_used)
    assert is_memory_byte_pair_or_none(total, used) is valid


@pytest.mark.parametrize("value", [True, False, 0.5, "25"])
def test_validate_busy_threshold_rejects_non_plain_integers(value):
    with pytest.raises(
        ValueError, match="busy_threshold must be -1 or an integer between 0 and 100"
    ):
        validate_busy_threshold(value)


@pytest.mark.parametrize(
    "value",
    [None, "job-123", "job_123", "job.123", "job~123", str(uuid.uuid4())],
)
def test_validate_job_id_accepts_omitted_and_url_path_safe_strings(value):
    assert session_config.validate_job_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "\t",
        123,
        True,
        ["job-123"],
        "job/123",
        "job?123",
        "job#123",
        "job 123",
        " job-123",
        "job-123 ",
        "job%123",
        "job:123",
    ],
)
def test_validate_job_id_rejects_invalid_custom_ids(value):
    with pytest.raises(ValueError, match="job_id"):
        session_config.validate_job_id(value)
