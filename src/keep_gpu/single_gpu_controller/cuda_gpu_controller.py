import asyncio
import random
import re
import subprocess
import time

import torch
from torch.multiprocessing.spawn import spawn

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import ComputingPlatform

logger = setup_logger(__name__)


class CudaGPUController(BaseGPUController):
    """Keep a single CUDA GPU busy with lightweight matrix multiplications.

    Parameters
    ----------
    rank : int
        Local GPU index handled by this controller.
    interval : float, default 1.0
        Sleep interval (seconds) between workload batches.
    matmul_iterations : int, default 5000
        Number of matmul operations per batch.
    vram_to_keep : int, default 0
        Target MB of free VRAM to keep (reserved for future use).
    busy_threshold : int, default 10
        If current GPU utilisation (%) exceeds this value, controller will rest.
    """

    def __init__(
        self,
        *,
        rank: int,
        interval: float = 1.0,
        matmul_iterations: int = 5000,
        vram_to_keep: int = 0,
        busy_threshold: int = 10,
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.rank = rank
        self.interval = interval
        self.matmul_iterations = matmul_iterations
        self.busy_threshold = busy_threshold
        self.platform = ComputingPlatform.CUDA

        torch.cuda.set_device(rank)
        logger.info(
            "rank %s: controller initialised (interval=%.2fs, iters=%d)",
            rank,
            interval,
            matmul_iterations,
        )

    # ------------------------------------------------------------------
    # Monitoring helpers
    # ------------------------------------------------------------------

    def monitor(self) -> int:
        """Return current GPU utilisation (%) using `nvidia-smi`."""
        proc = subprocess.Popen(
            ["nvidia-smi", "-i", str(self.rank)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = proc.communicate()
        for line in stdout.decode().split("\n")[::-1]:
            if "Default" in line:
                try:
                    return int(re.findall(r"\d+", line)[-1])
                except (IndexError, ValueError):
                    break
        logger.warning("rank %s: unable to parse nvidia-smi output", self.rank)
        return 0

    # ------------------------------------------------------------------
    # Main workload loop (sync & async)
    # ------------------------------------------------------------------

    def keep(self):
        """Blocking loop: run workloads, rest when busy, repeat."""
        logger.info("rank %s: entering keep loop", self.rank)
        while True:
            try:
                self._run_matmul_batch()
                self.rest()
                while self.monitor() > self.busy_threshold:
                    logger.warning(
                        "rank %s: GPU busy (> %d%%) – resting",
                        self.rank,
                        self.busy_threshold,
                    )
                    self.rest()
            except KeyboardInterrupt:
                logger.info("rank %s: interrupted by user – exiting", self.rank)
                break
            except RuntimeError as e:
                logger.error("rank %s: runtime error – %s", self.rank, e)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                self.rest()
            except Exception:
                logger.exception("rank %s: unexpected error", self.rank)
                self.rest()

    async def _keep(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.keep)

    # ------------------------------------------------------------------
    # Rest utilities
    # ------------------------------------------------------------------

    def rest(self):
        logger.debug("rank %s: resting for %.2fs", self.rank, self.interval)
        time.sleep(self.interval)

    async def _rest(self):
        await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    # Release utilities
    # ------------------------------------------------------------------
    def release(self):
        """Release resources and exit the controller."""
        logger.info("rank %s: releasing resources and exiting", self.rank)
        torch.cuda.empty_cache()
        # Additional cleanup can be added here if needed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_matmul_batch(self):
        n = random.randint(5, 9)
        a = torch.rand((8192 * n, 4096), device="cuda")
        b = torch.rand((4096, 8192 * 5), device="cuda")

        tic = time.time()
        for _ in range(self.matmul_iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        toc = time.time()

        logger.info(
            "rank %s: matmul batch done – avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / self.matmul_iterations,
        )


# ---------------------------------------------------------------------
# Multiprocessing launcher helpers
# ---------------------------------------------------------------------


def _worker(
    rank: int,
    interval: float,
    matmul_iterations: int,
    busy_threshold: int,
    vram_to_keep: int,
):
    controller = CudaGPUController(
        rank=rank,
        interval=interval,
        matmul_iterations=matmul_iterations,
        busy_threshold=busy_threshold,
        vram_to_keep=vram_to_keep,
    )
    controller.keep()


def run_benchmark(
    *,
    gpus: int = 1,
    interval: float = 1.0,
    matmul_iterations: int = 5000,
    busy_threshold: int = 10,
    vram_to_keep: int = 0,
):
    """Spawn *gpus* worker processes, each pinning to a distinct GPU index."""
    spawn(
        _worker,
        args=(interval, matmul_iterations, busy_threshold, vram_to_keep),
        nprocs=gpus,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CUDA GPU controller benchmark")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Sleep interval in seconds"
    )
    parser.add_argument(
        "--matmul_iterations",
        type=int,
        default=5000,
        help="Number of matmul iterations per batch",
    )
    parser.add_argument(
        "--busy_threshold", type=int, default=10, help="GPU busy threshold (%)"
    )
    parser.add_argument(
        "--vram_to_keep", type=int, default=0, help="VRAM to keep free (MB)"
    )

    args = parser.parse_args()
    run_benchmark(
        gpus=args.gpus,
        interval=args.interval,
        matmul_iterations=args.matmul_iterations,
        busy_threshold=args.busy_threshold,
        vram_to_keep=args.vram_to_keep,
    )
