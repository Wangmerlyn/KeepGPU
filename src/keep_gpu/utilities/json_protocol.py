"""Strict JSON helpers for public protocol boundaries."""

import json
from typing import Any


def _reject_nonstandard_json_constant(constant: str) -> None:
    raise ValueError(f"invalid JSON constant: {constant}")


def strict_json_loads(data: Any) -> Any:
    """Decode standard JSON while rejecting NaN and Infinity constants."""
    return json.loads(data, parse_constant=_reject_nonstandard_json_constant)
