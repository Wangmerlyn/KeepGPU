import re
from typing import Union

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


def _bytes_to_float32_elements(byte_count: float) -> int:
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
        return _bytes_to_float32_elements(float(value))
    if len(unit) > 1:
        # Treat all-lowercase units as byte units ("gb" -> "GB", "gib" -> "GIB")
        # while preserving explicit mixed-case bit forms ("Gb", "GIb").
        unit = unit.upper() if unit.islower() else unit[:-1].upper() + unit[-1]
    if unit not in _UNITS:
        raise ValueError(f"unknown unit: {unit}, should be one of {_UNITS.keys()}")
    return _bytes_to_float32_elements(float(value) * _UNITS[unit])


def parse_vram_to_elements(value: Union[int, str]) -> int:
    """Normalize public VRAM input to internal float32 element count."""
    if isinstance(value, str):
        return parse_size(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _bytes_to_float32_elements(float(value))
    raise TypeError(f"vram_to_keep must be str or int bytes, got {type(value)}")
