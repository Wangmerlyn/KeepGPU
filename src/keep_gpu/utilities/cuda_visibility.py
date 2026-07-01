from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

_MAX_DEVICE_ORDINAL_DIGITS = len(str(2**63 - 1))


@dataclass(frozen=True)
class CudaVisibleMask:
    tokens: Optional[Tuple[str, ...]]
    invalid: bool = False

    @property
    def permits_torch_fallback(self) -> bool:
        return not self.invalid and (self.tokens is None or bool(self.tokens))


def is_cuda_visible_index_token(token: str) -> bool:
    return token.isascii() and token.isdigit()


def cuda_visible_index_value(token: str) -> int | None:
    if not is_cuda_visible_index_token(token):
        return None
    normalized = token.lstrip("0") or "0"
    if len(normalized) > _MAX_DEVICE_ORDINAL_DIGITS:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def cuda_visible_token_key(token: str) -> tuple[str, int | str] | None:
    index_value = cuda_visible_index_value(token)
    if index_value is not None:
        return ("index", index_value)
    if is_cuda_visible_index_token(token):
        return None
    return ("token", token.lower())


def cuda_visible_mask() -> CudaVisibleMask:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return CudaVisibleMask(tokens=None)
    if raw.strip() in {"", "-1"}:
        return CudaVisibleMask(tokens=())

    parsed_tokens = []
    for token in (token.strip() for token in raw.split(",")):
        if token == "-1":
            break
        parsed_tokens.append(token)
    tokens = tuple(parsed_tokens)
    if any(not token or not token.isascii() for token in tokens):
        return CudaVisibleMask(tokens=(), invalid=True)

    token_keys = []
    for token in tokens:
        token_key = cuda_visible_token_key(token)
        if token_key is None:
            return CudaVisibleMask(tokens=(), invalid=True)
        token_keys.append(token_key)
    if len(set(token_keys)) != len(token_keys):
        return CudaVisibleMask(tokens=(), invalid=True)
    return CudaVisibleMask(tokens=tokens)


def decode_nvml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def lookup_nvml_uuid_handle(
    nvml_module,
    token: str,
    *,
    should_reraise: Callable[[Exception], bool] | None = None,
):
    uuid_lookup = getattr(nvml_module, "nvmlDeviceGetHandleByUUID", None)
    fallback_to_prefix = False
    if uuid_lookup is not None:
        for uuid in (token, token.encode("utf-8")):
            try:
                return uuid_lookup(uuid)
            except Exception as exc:
                if should_reraise is not None and should_reraise(exc):
                    raise
                if not isinstance(exc, (AttributeError, TypeError)):
                    fallback_to_prefix = True
                continue

    if not fallback_to_prefix:
        return None
    return lookup_nvml_uuid_prefix_handle(
        nvml_module,
        token,
        should_reraise=should_reraise,
    )


def lookup_nvml_uuid_prefix_handle(
    nvml_module,
    token: str,
    *,
    should_reraise: Callable[[Exception], bool] | None = None,
):
    count_lookup = getattr(nvml_module, "nvmlDeviceGetCount", None)
    handle_lookup = getattr(nvml_module, "nvmlDeviceGetHandleByIndex", None)
    uuid_lookup = getattr(nvml_module, "nvmlDeviceGetUUID", None)
    if count_lookup is None or handle_lookup is None or uuid_lookup is None:
        return None
    try:
        count = int(count_lookup())
    except Exception as exc:
        if should_reraise is not None and should_reraise(exc):
            raise
        return None
    if count < 0:
        return None

    matches = []
    normalized_token = token.lower()
    for index in range(count):
        try:
            handle = handle_lookup(index)
            uuid = decode_nvml_text(uuid_lookup(handle)).lower()
        except Exception as exc:
            if should_reraise is not None and should_reraise(exc):
                raise
            return None
        if uuid.startswith(normalized_token):
            matches.append(handle)
    return matches[0] if len(matches) == 1 else None


__all__ = [
    "CudaVisibleMask",
    "cuda_visible_index_value",
    "cuda_visible_mask",
    "cuda_visible_token_key",
    "decode_nvml_text",
    "is_cuda_visible_index_token",
    "lookup_nvml_uuid_handle",
    "lookup_nvml_uuid_prefix_handle",
]
