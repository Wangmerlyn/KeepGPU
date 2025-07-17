import torch

import threading
from typing import Optional, List

from keep_gpu.utilities.platform_manager import get_platform, ComputingPlatform


class GlobalGPUController:
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        interval: int = 300,
        vram_to_keep: int = 10 * (2**30),
    ):
        self.computing_platform = get_platform()
        if gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count()))
        self.interval = interval
        self.vram_to_keep = vram_to_keep
        if self.computing_platform == ComputingPlatform.CUDA:
            from keep_gpu.single_gpu_controller.cuda_gpu_controller import (
                CudaGPUController,
            )

            self.controllers = [
                CudaGPUController(rank=i, interval=interval, vram_to_keep=vram_to_keep)
                for i in self.gpu_ids
            ]
        else:
            raise NotImplementedError(
                f"GlobalGPUController not implemented for platform {self.computing_platform}"
            )

    def keep(self) -> None:
        for ctrl in self.controllers:
            ctrl.keep()

    def release(self) -> None:
        threads = []
        for ctrl in self.controllers:
            t = threading.Thread(target=ctrl.release)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def __enter__(self) -> "GlobalGPUController":
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
