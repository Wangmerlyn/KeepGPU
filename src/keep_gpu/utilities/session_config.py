from typing import Any, List, Optional, Union


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
    if len(gpu_ids) > 64:
        raise ValueError("gpu_ids has too many items")
    if any(not _is_plain_int(gpu_id) or gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("gpu_ids must contain non-negative integers")
    return list(gpu_ids)


def validate_interval(interval: Any) -> Union[int, float]:
    """Validate public interval input in seconds."""
    if not _is_plain_number(interval) or interval <= 0:
        raise ValueError("interval must be positive")
    return interval


def validate_busy_threshold(busy_threshold: Any) -> int:
    """Validate utilization threshold; -1 disables utilization backoff."""
    if not _is_plain_int(busy_threshold) or (
        busy_threshold != -1 and not 0 <= busy_threshold <= 100
    ):
        raise ValueError("busy_threshold must be -1 or an integer between 0 and 100")
    return busy_threshold
