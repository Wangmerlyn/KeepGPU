from enum import Enum
from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


class ComputingPlatform(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


def _check_cuda():
    # NOTE: This function checks for CUDA availability by trying to import pynvml
    # from nvidia-ml-py (the maintained NVML bindings).
    # See https://github.com/vllm-project/vllm/blob/536fd330036b0406786c847f68e4f67cba06f421/vllm/platforms/__init__.py#L58
    # for related discussion.
    try:
        import pynvml

        pynvml.nvmlInit()
        return ComputingPlatform.CUDA
    except Exception as e:
        logger.debug(f"CUDA not available: {e}")
        return None


def _check_rocm():
    try:
        import rocm_smi

        rocm_smi.rocm_smi_init()
        return ComputingPlatform.ROCM
    except Exception as e:
        logger.debug(f"ROCM not available: {e}")
        return None


def _check_cpu():
    return ComputingPlatform.CPU


platform_check_funcs = {
    ComputingPlatform.CUDA: _check_cuda,
    ComputingPlatform.ROCM: _check_rocm,
    ComputingPlatform.CPU: _check_cpu,
}


def get_platform():
    """
    Return the current computing platform.
    """
    for platform, check_func in platform_check_funcs.items():
        if check_func() is not None:
            logger.info(f"Detected computing platform: {platform.value}")
            return platform
    logger.info("No specific computing platform detected, defaulting to CPU.")
    return ComputingPlatform.CPU  # Default to CPU if no other platform is available


if __name__ == "__main__":

    print("Current platform:", get_platform().value)
