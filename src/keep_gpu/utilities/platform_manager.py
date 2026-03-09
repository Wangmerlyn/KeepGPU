import os
import sys
import platform
from enum import Enum
from typing import Callable, List, Tuple

import torch

from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


class ComputingPlatform(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MACM = "macm"


def _check_cuda():
    """
    Return True if CUDA appears available.

    - Prefer torch reporting CUDA with no ROCm build.
    - Fall back to NVML availability.
    """
    try:
        # ROCm builds set torch.version.hip; treat those as non-CUDA.
        if torch.cuda.is_available() and torch.version.hip is None:
            return True
    except Exception as exc:  # pragma: no cover - torch edge cases
        logger.debug("torch.cuda.is_available() failed: %s", exc)

    try:
        import pynvml  # provided by nvidia-ml-py

        pynvml.nvmlInit()
        return True
    except Exception as exc:
        logger.debug("NVML unavailable: %s", exc)
        return False


def _check_rocm():
    try:
        if torch.cuda.is_available() and torch.version.hip:
            return True
    except Exception as exc:  # pragma: no cover - torch edge cases
        logger.debug("torch ROCm detection failed: %s", exc)

    try:
        import rocm_smi

        rocm_smi.rocm_smi_init()
        return True
    except Exception as exc:
        logger.debug("ROCm SMI unavailable: %s", exc)
        return False


def _check_cpu():
    return True


def _check_macm():
    """Return True if running on Apple Silicon Mac (Mac M) with MPS support."""
    try:
        # macOS (darwin) on Apple Silicon (arm64) with PyTorch MPS backend available
        if sys.platform != "darwin":
            return False
        if platform.machine() != "arm64":
            return False
        # PyTorch MPS availability
        if torch.backends.mps.is_available():
            return True
        return False
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("MACM detection failed: %s", exc)
        return False


_PLATFORM_CHECKS: List[Tuple[ComputingPlatform, Callable[[], bool]]] = [
    (ComputingPlatform.CUDA, _check_cuda),
    (ComputingPlatform.ROCM, _check_rocm),
    (ComputingPlatform.MACM, _check_macm),
    (ComputingPlatform.CPU, _check_cpu),
]

_cached_platform: ComputingPlatform | None = None


def get_platform():
    """
    Return the current computing platform.
    """
    global _cached_platform

    if _cached_platform is not None:
        return _cached_platform

    override = os.getenv("KEEP_GPU_PLATFORM")
    if override:
        try:
            platform = ComputingPlatform(override.lower())
            logger.info("Using KEEP_GPU_PLATFORM=%s override", platform.value)
            _cached_platform = platform
            return platform
        except ValueError:
            logger.warning(
                "Invalid KEEP_GPU_PLATFORM=%s; falling back to auto-detect", override
            )

    for platform, check_func in _PLATFORM_CHECKS:
        try:
            if check_func():
                logger.info("Detected computing platform: %s", platform.value)
                _cached_platform = platform
                return platform
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Platform check %s failed: %s", platform.value, exc)

    logger.info("No specific platform detected, defaulting to CPU.")
    _cached_platform = ComputingPlatform.CPU
    return _cached_platform  # Default to CPU if no other platform is available


if __name__ == "__main__":
    print("Current platform:", get_platform().value)
