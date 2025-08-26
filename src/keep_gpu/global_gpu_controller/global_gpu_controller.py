import threading
from typing import List, Optional, Union

import torch

from keep_gpu.utilities.humanized_input import parse_size
from keep_gpu.utilities.platform_manager import ComputingPlatform, get_platform


class GlobalGPUController:
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        interval: int = 300,
        vram_to_keep: Union[int, str] = 10 * (2**30),
        busy_threshold: int = 10,
    ):
        self.computing_platform = get_platform()
        self.interval = interval
        self.vram_to_keep = vram_to_keep
        if isinstance(self.vram_to_keep, str):
            try:
                self.vram_to_keep = parse_size(self.vram_to_keep)
            except ValueError as e:
                raise ValueError(
                    f"Invalid vram_to_keep value: {self.vram_to_keep}. Must be an integer (bytes) or a string like '1GiB', '2MiB' etc."
                ) from e
        if self.computing_platform == ComputingPlatform.CUDA:
            from keep_gpu.single_gpu_controller.cuda_gpu_controller import (
                CudaGPUController,
            )

            if gpu_ids is None:
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = gpu_ids

            self.controllers = [
                CudaGPUController(
                    rank=i,
                    interval=interval,
                    vram_to_keep=vram_to_keep,
                    busy_threshold=busy_threshold,
                )
                for i in self.gpu_ids
            ]
        else:
            raise NotImplementedError(
                f"GlobalGPUController not implemented for platform {self.computing_platform}"
            )

    def keep(self) -> None:
        for ctrl in self.controllers:
            ctrl.keep()

    @staticmethod
    def parse_size(text: str) -> int:
        return parse_size(text)

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
