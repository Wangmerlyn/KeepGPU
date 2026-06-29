from __future__ import annotations

import os
from typing import Optional, Tuple

_UNSET = object()


def _is_ascii_digit_token(token: str) -> bool:
    return token.isascii() and token.isdigit()


def _parse_numeric_mask(name: str) -> Optional[Tuple[int, ...]]:
    raw = os.environ.get(name)
    if raw is None:
        return None

    tokens = [token.strip() for token in raw.split(",")]
    if not tokens or any(
        not token or not _is_ascii_digit_token(token) for token in tokens
    ):
        return ()

    values = tuple(int(token) for token in tokens)
    if len(set(values)) != len(values):
        return ()
    return values


def rocm_monitor_device_count(rocm_smi) -> Optional[int]:
    count_fn = getattr(rocm_smi, "rsmi_num_monitor_devices", None)
    if count_fn is None:
        return None
    # ROCm SMI probes can fail with vendor-specific errors.
    try:
        count = int(count_fn())
    except Exception:  # noqa: BLE001
        return None
    return count if count >= 0 else None


def _physical_ids_available(
    physical_ids: Tuple[int, ...],
    monitor_count: Optional[int],
) -> bool:
    if monitor_count is None:
        return True
    return all(physical_id < monitor_count for physical_id in physical_ids)


def _effective_overlay_mask() -> Optional[Tuple[int, ...]]:
    hip_mask = _parse_numeric_mask("HIP_VISIBLE_DEVICES")
    cuda_mask = _parse_numeric_mask("CUDA_VISIBLE_DEVICES")

    if hip_mask == () or cuda_mask == ():
        return ()
    if hip_mask is not None and cuda_mask is not None:
        return hip_mask if hip_mask == cuda_mask else ()
    return hip_mask if hip_mask is not None else cuda_mask


def resolve_rocm_visible_rank_to_smi_index(
    visible_rank: int,
    rocm_smi=None,
    *,
    monitor_count=_UNSET,
) -> Optional[int]:
    if visible_rank < 0:
        return None

    if monitor_count is _UNSET:
        monitor_count = rocm_monitor_device_count(rocm_smi)

    base_mask = _parse_numeric_mask("ROCR_VISIBLE_DEVICES")
    overlay_mask = _effective_overlay_mask()
    if base_mask == () or overlay_mask == ():
        return None

    if base_mask is None and overlay_mask is None:
        if monitor_count is not None and visible_rank >= monitor_count:
            return None
        return visible_rank

    if base_mask is not None:
        if not _physical_ids_available(base_mask, monitor_count):
            return None
        if overlay_mask is None:
            visible_physical_ids = base_mask
        else:
            if any(rank >= len(base_mask) for rank in overlay_mask):
                return None
            visible_physical_ids = tuple(base_mask[rank] for rank in overlay_mask)
    else:
        assert overlay_mask is not None
        visible_physical_ids = overlay_mask
        if not _physical_ids_available(visible_physical_ids, monitor_count):
            return None

    if visible_rank >= len(visible_physical_ids):
        return None
    return visible_physical_ids[visible_rank]


__all__ = [
    "resolve_rocm_visible_rank_to_smi_index",
    "rocm_monitor_device_count",
]
