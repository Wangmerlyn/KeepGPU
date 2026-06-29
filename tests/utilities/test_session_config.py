import math
import uuid

import pytest

from keep_gpu.utilities import session_config
from keep_gpu.utilities.session_config import (
    PUBLIC_INTERVAL_MAX_SECONDS,
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
