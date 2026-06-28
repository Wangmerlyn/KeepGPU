import math
import re
from decimal import Decimal, InvalidOperation
from typing import Union

PUBLIC_VRAM_MAX_BYTES = 1 << 50
_PUBLIC_VRAM_MAX_BYTES_TEXT = str(PUBLIC_VRAM_MAX_BYTES)
_PUBLIC_VRAM_MAX_LABEL = "1 PiB"

_UNITS = {
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "Kb": 1000 / 8,
    "Mb": 1000**2 / 8,
    "Gb": 1000**3 / 8,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "KIb": 1024 / 8,
    "MIb": 1024**2 / 8,
    "GIb": 1024**3 / 8,
}


def _vram_too_large_message() -> str:
    return f"vram must be no more than {_PUBLIC_VRAM_MAX_LABEL}"


def _parse_public_number(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"invalid format: {value}, should be like '1000 MB'") from exc
    except ValueError as exc:
        raise ValueError(_vram_too_large_message()) from exc


def _parse_public_byte_count(value: str) -> int:
    normalized = value.lstrip("0") or "0"
    if len(normalized) > len(_PUBLIC_VRAM_MAX_BYTES_TEXT) or (
        len(normalized) == len(_PUBLIC_VRAM_MAX_BYTES_TEXT)
        and normalized > _PUBLIC_VRAM_MAX_BYTES_TEXT
    ):
        raise ValueError(_vram_too_large_message())
    return int(normalized)


def _unit_factor(unit: str) -> Decimal:
    return Decimal(str(_UNITS[unit]))


def _bytes_to_float32_elements(byte_count: Union[int, float, Decimal]) -> int:
    if isinstance(byte_count, float) and not math.isfinite(byte_count):
        raise ValueError(_vram_too_large_message())
    if byte_count > PUBLIC_VRAM_MAX_BYTES:
        raise ValueError(_vram_too_large_message())
    if isinstance(byte_count, int):
        elements = byte_count // 4
    elif isinstance(byte_count, Decimal):
        elements = int(byte_count // Decimal(4))
    else:
        elements = int(byte_count / 4)
    if elements <= 0:
        raise ValueError("memory size must be at least 4 bytes")
    return elements


def parse_size(text: str) -> int:
    """
    Parse human-readable memory strings into float32 element counts.

    The return value is the number of float32 elements needed to occupy the
    requested memory size. When no unit is provided, the value is interpreted
    as raw bytes. Supported units are the keys in `_UNITS`.
    """
    text = text.strip().replace(" ", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([A-Za-z]*)", text)
    if not m:
        raise ValueError(f"invalid format: {text}, should be like '1000 MB'")
    value, unit = m.groups()
    if not unit:
        if value.isdigit():
            return _bytes_to_float32_elements(_parse_public_byte_count(value))
        return _bytes_to_float32_elements(_parse_public_number(value))
    if len(unit) > 1:
        # Treat all-lowercase units as byte units ("gb" -> "GB", "gib" -> "GIB")
        # while preserving explicit mixed-case bit forms ("Gb", "GIb").
        unit = unit.upper() if unit.islower() else unit[:-1].upper() + unit[-1]
    if unit not in _UNITS:
        raise ValueError(f"unknown unit: {unit}, should be one of {_UNITS.keys()}")
    number = _parse_public_number(value)
    factor = _unit_factor(unit)
    if number > Decimal(PUBLIC_VRAM_MAX_BYTES) / factor:
        raise ValueError(_vram_too_large_message())
    return _bytes_to_float32_elements(number * factor)


def parse_vram_to_elements(value: Union[int, str]) -> int:
    """Normalize public VRAM input to internal float32 element count."""
    if isinstance(value, str):
        return parse_size(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _bytes_to_float32_elements(value)
    raise TypeError(f"vram_to_keep must be str or int bytes, got {type(value)}")
