from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple


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

    tokens = tuple(token.strip() for token in raw.split(","))
    if any(not token or token == "-1" or not token.isascii() for token in tokens):
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


__all__ = [
    "CudaVisibleMask",
    "cuda_visible_index_value",
    "cuda_visible_mask",
    "cuda_visible_token_key",
    "is_cuda_visible_index_token",
]
