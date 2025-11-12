import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.gpu_monitor import get_gpu_utilization
from keep_gpu.utilities.humanized_input import parse_size
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import ComputingPlatform

logger = setup_logger(__name__)


class CudaGPUController(BaseGPUController):
    """CudaGPUController
    Keep a single CUDA GPU busy by repeatedly running lightweight
    matrix-multiplication workloads in a background thread.

    Typical usage:

    ```python
    ctrl = CudaGPUController(rank=0, interval=0.5)
    ctrl.keep()          # occupy GPU while you do CPU-only work
    dataset.process()
    ctrl.release()        # give GPU memory back
    model.train_start()   # now run real GPU training
    ```

    Or as a context manager:

    ```python
    with CudaGPUController(rank=0, interval=0.5):
        dataset.process()  # GPU occupied inside this block
    model.train_start()    # GPU free after exiting block
    ```
    """

    def __init__(
        self,
        *,
        rank: int,
        interval: float = 1.0,
        matmul_iterations: int = 5000,
        vram_to_keep: str | int = "1000 MB",
        busy_threshold: int = 10,
    ):
        """
        Args:
            rank (int): Local CUDA device index to occupy.
            interval (float, optional): Sleep time (seconds) between workload
                batches. Defaults to 0.5.
            matmul_iterations (int, optional): Number of matmul ops per batch.
            vram_to_keep (int or str, optional): Amount of VRAM to keep busy,
                e.g. `"1000 MB"`, `"20 GB"`, or an integer like `1000 * 1000`.
                This represents the total size of the matrix allocated to
                occupy the GPU.
            busy_threshold (int, optional): If current utilisation (%) exceeds
                this threshold, the worker will insert extra sleeps to avoid
                hogging the GPU.

        """
        if isinstance(vram_to_keep, str):
            vram_to_keep = self.parse_size(vram_to_keep)
        elif isinstance(vram_to_keep, int):
            vram_to_keep = vram_to_keep
        else:
            raise TypeError(
                f"vram_to_keep must be str or int, got {type(vram_to_keep)}"
            )
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.interval = interval
        self.matmul_iterations = matmul_iterations
        self.busy_threshold = busy_threshold
        self.platform = ComputingPlatform.CUDA

        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def parse_size(text: str) -> int:
        return parse_size(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keep(self) -> None:
        """Launch the background thread that keeps the GPU busy."""
        if self._thread and self._thread.is_alive():
            logger.warning("rank %s: keep thread already running", self.rank)
            return

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._keep_loop,
            name=f"gpu-keeper-{self.rank}",
            daemon=True,  # daemon so program can exit cleanly
        )
        self._thread.start()
        logger.info("rank %s: keep thread started", self.rank)

    def release(self) -> None:
        """
        Stop the background thread and clear CUDA cache so the memory
        becomes immediately available to other code.
        """
        if not (self._thread and self._thread.is_alive()):
            logger.warning("rank %s: keep thread not running", self.rank)
            return

        self._stop_evt.set()
        self._thread.join()
        torch.cuda.empty_cache()
        logger.info("rank %s: keep thread stopped & cache cleared", self.rank)

    # Context-manager helpers -------------------------------------------------
    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------
    def _keep_loop(self) -> None:
        """Internal: run workloads until stop event is set."""
        torch.cuda.set_device(self.rank)
        matrix = None
        while not self._stop_evt.is_set():
            try:
                matrix = torch.rand(
                    self.vram_to_keep,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                break
            except RuntimeError as e:
                logger.error("rank %s: failed to allocate matrix: %s", self.rank, e)
                time.sleep(self.interval)
        if matrix is None:
            logger.error("rank %s: failed to allocate matrix, exiting loop", self.rank)
            raise RuntimeError("Failed to allocate matrix for GPU keeping")
        while not self._stop_evt.is_set():
            try:
                gpu_utilization = self._monitor_utilization(self.rank)
                if gpu_utilization > self.busy_threshold:
                    logger.debug(
                        "rank %s: GPU busy (%d%%), sleeping longer",
                        self.rank,
                        gpu_utilization,
                    )
                else:
                    self._run_mat_batch(matrix)
                time.sleep(self.interval)
            except RuntimeError as e:
                # Handle OOM by clearing cache; then sleep and continue
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                time.sleep(self.interval)
            except Exception:
                # Log unexpected exceptions but keep running
                logger.exception("rank %s: unexpected error", self.rank)
                time.sleep(self.interval)

    # ------------------------------------------------------------------
    # Workload implementation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_mat_batch(self, matrix: torch.Tensor) -> None:
        """Run a batch of dummy matmuls to keep GPU busy."""

        tic = time.time()
        for _ in range(self.matmul_iterations):
            torch.relu_(matrix)
            if self._stop_evt.is_set():
                break
        torch.cuda.synchronize()
        toc = time.time()

        logger.debug(
            "rank %s: mat ops batch done – avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / self.matmul_iterations,
        )

    # ------------------------------------------------------------------
    # Utilization monitor
    # ------------------------------------------------------------------
    @staticmethod
    def _monitor_utilization(rank: int) -> int:
        """
        Return current GPU utilization (%) for `rank`.
        Falls back to 0 when NVML is unavailable.
        """
        utilization = get_gpu_utilization(rank)
        return utilization if utilization is not None else 0
