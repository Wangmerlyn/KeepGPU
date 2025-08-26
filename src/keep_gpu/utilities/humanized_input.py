import re

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


def parse_size(text: str) -> int:
    text = text.strip().replace(" ", "")
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([A-Za-z]*)", text)
    if not m:
        raise ValueError(f"invalid format: {text}, should be like '1000 MB'")
    value, unit = m.groups()
    unit = unit or "GB"
    if len(unit) > 1:
        unit = unit[:-1].upper() + unit[-1]
    if unit not in _UNITS:
        raise ValueError(f"unknown unit: {unit}, should be one of {_UNITS.keys()}")
    return int(float(value) * _UNITS[unit] / 4)  # float32 takes 4 bytes
