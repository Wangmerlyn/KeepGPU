from __future__ import annotations

from typing import Any, Dict, List

import torch

from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


def _query_nvml() -> List[Dict[str, Any]]:
    import pynvml

    pynvml.nvmlInit()
    infos: List[Dict[str, Any]] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            infos.append(
                {
                    "id": idx,
                    "platform": "cuda",
                    "name": name,
                    "memory_total": int(mem.total),
                    "memory_used": int(mem.used),
                    "utilization": int(util),
                }
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
    try:
        rocm_smi.rsmi_init()
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
                    "platform": "rocm",
                    "name": name,
                    "memory_total": int(total) if total is not None else None,
                    "memory_used": int(used) if used is not None else None,
                    "utilization": util,
                }
            )
    finally:
        try:
            rocm_smi.rsmi_shut_down()
        except Exception:
            pass
    return infos


def _query_torch() -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    if not torch.cuda.is_available():
        return infos
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
                    "platform": "cuda" if torch.version.hip is None else "rocm",
                    "name": name,
                    "memory_total": int(total) if total is not None else None,
                    "memory_used": int(used) if used is not None else None,
                    "utilization": None,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Torch GPU info failed: %s", exc)
    return infos


def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Return a list of GPU info dicts: id, platform, name, memory_total, memory_used, utilization.
    Tries NVML first (CUDA), then ROCm SMI, then falls back to torch.cuda data.
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

    return _query_torch()


__all__ = ["get_gpu_info"]
