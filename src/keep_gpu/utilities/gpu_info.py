from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


def _decode_nvml_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return str(value)


def _cuda_visible_tokens() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    if raw.strip() in {"", "-1"}:
        return []
    tokens = [token.strip() for token in raw.split(",")]
    if any(not token or token == "-1" for token in tokens):
        return []
    if len(set(tokens)) != len(tokens):
        return []
    return tokens


def _lookup_nvml_uuid_handle(pynvml, token: str):
    uuid_lookup = getattr(pynvml, "nvmlDeviceGetHandleByUUID", None)
    if uuid_lookup is None:
        return None

    for uuid in (token, token.encode("utf-8")):
        try:
            return uuid_lookup(uuid)
        except Exception:
            continue
    return None


def _nvml_physical_id(pynvml, handle, fallback: Optional[int] = None) -> Optional[int]:
    if fallback is not None:
        return fallback

    index_lookup = getattr(pynvml, "nvmlDeviceGetIndex", None)
    if index_lookup is None:
        return None
    try:
        return int(index_lookup(handle))
    except Exception:
        return None


def _nvml_uuid(pynvml, handle) -> Optional[str]:
    uuid_lookup = getattr(pynvml, "nvmlDeviceGetUUID", None)
    if uuid_lookup is None:
        return None
    try:
        return _decode_nvml_text(uuid_lookup(handle))
    except Exception:
        return None


def _nvml_info_for_handle(
    pynvml,
    handle,
    visible_id: int,
    physical_id: Optional[int] = None,
) -> Dict[str, Any]:
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    name = _decode_nvml_text(pynvml.nvmlDeviceGetName(handle))
    info: Dict[str, Any] = {
        "id": visible_id,
        "visible_id": visible_id,
        "platform": "cuda",
        "name": name,
        "memory_total": int(mem.total),
        "memory_used": int(mem.used),
        "utilization": int(util),
    }
    resolved_physical_id = _nvml_physical_id(pynvml, handle, physical_id)
    if resolved_physical_id is not None:
        info["physical_id"] = resolved_physical_id
    uuid = _nvml_uuid(pynvml, handle)
    if uuid is not None:
        info["uuid"] = uuid
    return info


def _query_nvml() -> List[Dict[str, Any]]:
    import pynvml

    pynvml.nvmlInit()
    infos: List[Dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        visible_tokens = _cuda_visible_tokens()
        if visible_tokens is None:
            visible_tokens = [str(idx) for idx in range(count)]

        for token in visible_tokens:
            if token.isdigit() and int(token) >= count:
                return []

        visible_handles = []
        seen_physical_ids = set()
        for visible_id, token in enumerate(visible_tokens):
            physical_id = int(token) if token.isdigit() else None
            try:
                handle = (
                    pynvml.nvmlDeviceGetHandleByIndex(physical_id)
                    if physical_id is not None
                    else _lookup_nvml_uuid_handle(pynvml, token)
                )
            except Exception:
                handle = None

            if handle is None:
                return []

            resolved_physical_id = _nvml_physical_id(pynvml, handle, physical_id)
            if (
                resolved_physical_id is not None
                and resolved_physical_id in seen_physical_ids
            ):
                return []
            seen_physical_ids.add(resolved_physical_id)
            visible_handles.append((visible_id, handle, resolved_physical_id))

        for visible_id, handle, physical_id in visible_handles:
            infos.append(
                _nvml_info_for_handle(
                    pynvml,
                    handle,
                    visible_id=visible_id,
                    physical_id=physical_id,
                )
            )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return infos


def _query_rocm() -> List[Dict[str, Any]]:
    try:
        import rocm_smi  # type: ignore
    except Exception as exc:  # pragma: no cover - env-specific
        logger.debug("rocm_smi import failed: %s", exc)
        return []

    infos: List[Dict[str, Any]] = []
    current_device = None
    try:
        rocm_smi.rsmi_init()
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
        # Use torch to enumerate devices for names/memory
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for idx in range(count):
            util = None
            try:
                util = int(rocm_smi.rsmi_dev_busy_percent_get(idx))
            except Exception as exc:
                logger.debug("ROCm util query failed for %s: %s", idx, exc)

            try:
                torch.cuda.set_device(idx)
                free, total = torch.cuda.mem_get_info()
                used = total - free
            except Exception:
                total = used = None

            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = f"rocm:{idx}"

            infos.append(
                {
                    "id": idx,
                    "visible_id": idx,
                    "platform": "rocm",
                    "name": name,
                    "memory_total": int(total) if total is not None else None,
                    "memory_used": int(used) if used is not None else None,
                    "utilization": util,
                }
            )
    finally:
        if current_device is not None:
            try:
                torch.cuda.set_device(current_device)
            except Exception:
                pass
        try:
            rocm_smi.rsmi_shut_down()
        except Exception:
            pass
    return infos


def _query_torch() -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return infos
    current_device = torch.cuda.current_device()
    try:
        count = torch.cuda.device_count()
        for idx in range(count):
            torch.cuda.set_device(idx)
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
            except Exception:
                total = used = None
            try:
                name = torch.cuda.get_device_name(idx)
            except Exception:
                name = f"cuda:{idx}"
            infos.append(
                {
                    "id": idx,
                    "visible_id": idx,
                    "platform": "cuda" if torch.version.hip is None else "rocm",
                    "name": name,
                    "memory_total": int(total) if total is not None else None,
                    "memory_used": int(used) if used is not None else None,
                    "utilization": None,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Torch GPU info failed: %s", exc)
    finally:
        try:
            torch.cuda.set_device(current_device)
        except Exception:
            pass
    return infos


def _safe_mps_memory_value(method_name: str) -> int | None:
    try:
        mps = getattr(torch, "mps")
        method = getattr(mps, method_name)
        return int(method())
    except Exception as exc:
        logger.debug("MPS memory query %s failed: %s", method_name, exc)
        return None


def _query_mps() -> List[Dict[str, Any]]:
    try:
        if not torch.backends.mps.is_available():
            return []
    except Exception as exc:
        logger.debug("MPS availability query failed: %s", exc)
        return []

    current_allocated = _safe_mps_memory_value("current_allocated_memory")
    driver_allocated = _safe_mps_memory_value("driver_allocated_memory")
    recommended_max = _safe_mps_memory_value("recommended_max_memory")

    return [
        {
            "id": 0,
            "visible_id": 0,
            "platform": "macm",
            "name": "Apple Silicon GPU",
            "memory_total": recommended_max,
            "memory_used": (
                driver_allocated if driver_allocated is not None else current_allocated
            ),
            "utilization": None,
            "memory_allocated": current_allocated,
        }
    ]


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Return GPU info dicts where id is the public visible ordinal.

    Each record includes id, visible_id, platform, name, memory_total,
    memory_used, and utilization. Backends may add physical_id or uuid metadata
    when the underlying vendor identity is known.

    Tries NVML first (CUDA), then ROCm SMI, then torch.cuda, then MPS data.
    """
    try:
        infos = _query_nvml()
        if infos:
            return infos
    except Exception as exc:
        logger.debug("NVML info failed: %s", exc)

    try:
        infos = _query_rocm()
        if infos:
            return infos
    except Exception as exc:
        logger.debug("ROCm info failed: %s", exc)

    try:
        infos = _query_torch()
        if infos:
            return infos
    except Exception as exc:
        logger.debug("Torch GPU info failed: %s", exc)

    return _query_mps()


__all__ = ["get_gpu_info"]
