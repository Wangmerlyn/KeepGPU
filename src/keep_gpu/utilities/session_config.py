import math
import re
import threading
from typing import Any, List, Optional, Union

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9._~-]+$")
DEFAULT_BUSY_THRESHOLD = 25
PUBLIC_INTERVAL_MAX_SECONDS = int(threading.TIMEOUT_MAX)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_gpu_ids(gpu_ids: Any) -> Optional[List[int]]:
    """Validate public GPU id input and return a normalized list."""
    if gpu_ids is None:
        return None
    if not isinstance(gpu_ids, list):
        raise ValueError("gpu_ids must be a list of integers")
    if not gpu_ids:
        raise ValueError("gpu_ids must select at least one GPU")
    if len(gpu_ids) > 64:
        raise ValueError("gpu_ids has too many items")
    if any(not _is_plain_int(gpu_id) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must contain non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu_ids must not contain duplicate values")
    return list(gpu_ids)


def validate_interval(interval: Any) -> Union[int, float]:
    """Validate public interval input in seconds."""
    if not _is_plain_number(interval):
        raise ValueError("interval must be finite and positive")
    if isinstance(interval, float) and not math.isfinite(interval):
        raise ValueError("interval must be finite and positive")
    if interval <= 0:
        raise ValueError("interval must be positive")
    if interval > PUBLIC_INTERVAL_MAX_SECONDS:
        raise ValueError(
            f"interval must be no more than {PUBLIC_INTERVAL_MAX_SECONDS} seconds"
        )
    return interval


def validate_busy_threshold(busy_threshold: Any) -> int:
    """Validate utilization threshold; -1 disables utilization backoff."""
    if not _is_plain_int(busy_threshold) or (
        busy_threshold != -1 and not 0 <= busy_threshold <= 100
    ):
        raise ValueError("busy_threshold must be -1 or an integer between 0 and 100")
    return busy_threshold


def validate_positive_integer(value: Any, name: str) -> int:
    """Validate a public positive integer input."""
    if not _is_plain_int(value):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_visible_rank(rank: Any, visible_count: Any) -> int:
    """Validate a public single-GPU visible device ordinal."""
    if not _is_plain_int(rank):
        raise TypeError("rank must be an integer")
    if not _is_plain_int(visible_count) or visible_count < 0:
        raise ValueError("visible device count must be a non-negative integer")
    if visible_count == 0:
        raise ValueError("no visible GPUs are available; rank cannot be selected")
    if rank < 0 or rank >= visible_count:
        raise ValueError(
            "rank must be a visible device ordinal less than "
            f"{visible_count}; got {rank}"
        )
    return rank


def validate_job_id(job_id: Any) -> Optional[str]:
    """Validate public session job_id input."""
    if job_id is None:
        return None
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a URL-path-safe non-empty string")
    if not job_id.strip() or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("job_id must be a URL-path-safe non-empty string")
    return job_id
