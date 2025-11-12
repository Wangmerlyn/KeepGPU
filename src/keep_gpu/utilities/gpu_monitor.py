"""Utilities for querying GPU utilization in a Pythonic way."""

from __future__ import annotations

import atexit
import threading
from typing import Optional

from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)

try:  # pragma: no cover - import guard
    # Provided by the maintained `nvidia-ml-py` package.
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - env without NVML
    pynvml = None


class NVMLMonitor:
    """Lightweight wrapper around NVML to read GPU utilization."""

    def __init__(self, nvml_module) -> None:
        self._nvml = nvml_module
        self._lock = threading.Lock()
        self._initialized = False
        self._shutdown_registered = False

    def _ensure_initialized(self) -> bool:
        if self._nvml is None:
            return False
        if self._initialized:
            return True

        with self._lock:
            if self._initialized:
                return True
            try:
                self._nvml.nvmlInit()
            except Exception as exc:  # pragma: no cover - passthrough
                logger.debug("NVML init failed: %s", exc)
                return False

            if not self._shutdown_registered:
                atexit.register(self._safe_shutdown)
                self._shutdown_registered = True

            self._initialized = True
            return True

    def _safe_shutdown(self) -> None:
        if not self._nvml or not self._initialized:
            return
        try:
            self._nvml.nvmlShutdown()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("NVML shutdown failed: %s", exc)
        finally:
            self._initialized = False

    def get_gpu_utilization(self, index: int) -> Optional[int]:
        """Return utilization percentage for `index`, or None when unavailable."""
        if not self._ensure_initialized():
            return None

        try:
            handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
            rates = self._nvml.nvmlDeviceGetUtilizationRates(handle)
            return int(rates.gpu)
        except self._nvml.NVMLError as exc:
            logger.debug("NVML query failed for GPU %s: %s", index, exc)
            return None


_nvml_monitor = NVMLMonitor(pynvml)


def get_gpu_utilization(index: int) -> Optional[int]:
    """Return utilization percentage for `index`, or None when unavailable."""
    return _nvml_monitor.get_gpu_utilization(index)


__all__ = ["get_gpu_utilization", "NVMLMonitor"]
