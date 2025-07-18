import threading
import time
import torch
import subprocess
import re
from typing import Optional

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import ComputingPlatform

logger = setup_logger(__name__)


_UNITS = {
    'KB': 1000,  'MB': 1000**2,  'GB': 1000**3,
    'Kb': 1000 / 8, 'Mb': 1000**2 / 8, 'Gb': 1000**3 / 8,
    'KIB': 1024, 'MIB': 1024**2, 'GIB': 1024**3,
    'KIb': 1024 / 8, 'MIb': 1024**2 / 8, 'GIb': 1024**3 / 8,  
}


class CudaGPUController(BaseGPUController):
    """
    Keep a single CUDA GPU busy by repeatedly running lightweight
    matrix-multiplication workloads in a background thread.

    Typical usage pattern
    ---------------------
    >>> ctrl = CudaGPUController(rank=0, interval=0.5)
    >>> ctrl.start()          # occupy GPU while you do CPU-only work
    >>> dataset.process()
    >>> ctrl.release()        # give GPU memory back
    >>> model.train_start()   # now run real GPU training

    You can also use the controller as a context manager:

    >>> with CudaGPUController(rank=0, interval=0.5):
    ...     dataset.process()  # GPU occupied inside this block
    >>> model.train_start()    # GPU free after exiting block
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
        Parameters
        ----------
        rank : int
            Local CUDA device index to occupy.
        interval : float, optional
            Sleep time (seconds) between workload batches.
        matmul_iterations : int, optional
            Number of matmul ops per batch.
        vram_to_keep : str | int, optional
            Amount of VRAM to keep busy, e.g. "1000 MB", "20 GB" or 1000 * 1000.
            This is the total size of the matrix allocated to keep the GPU busy.
        busy_threshold : int, optional
            If current utilisation (%) exceeds this value, the worker will
            insert extra sleeps to avoid hogging the GPU.
        """
        if type(vram_to_keep) is str:
            vram_to_keep = self.parse_size(vram_to_keep)
        elif type(vram_to_keep) is int:
            vram_to_keep = vram_to_keep
        else:
            raise TypeError(f"vram_to_keep must be str or int, got {type(vram_to_keep)}")
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
        text = text.strip().replace(' ', '').upper()
        m = re.fullmatch(r'([0-9]*\.?[0-9]+)([A-Z]*)', text)
        if not m:
            raise ValueError(f"invalid format: {text}, should be like '1000 MB'")
        value, unit = m.groups()
        unit = unit or 'GB'
        if len(unit) > 1:
            unit = unit[:-1].upper() + unit[-1]
        if unit not in _UNITS:
            raise ValueError(f"unknown unit: {unit}, should be one of {_UNITS.keys()}")
        return int(float(value) * _UNITS[unit]/8)
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
        while not self._stop_evt.is_set():
            try:
                self._run_mat_batch()
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
    def _run_mat_batch(self) -> None:
        """Run a batch of dummy matmuls to keep GPU busy."""
        matrix = torch.rand(self.vram_to_keep, device=self.device)
        tic = time.time()
        for _ in range(self.matmul_iterations):
            torch.relu(matrix)
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
    # Optional: simple nvidia-smi monitor (not used in thread version)
    # ------------------------------------------------------------------
    @staticmethod
    def _monitor_utilization(rank: int) -> int:
        """
        Return current GPU utilization (%) for `rank`
        by parsing `nvidia-smi` output. Can be plugged into
        `_keep_loop` if you want adaptive sleeping.
        """
        proc = subprocess.Popen(
            ["nvidia-smi", "-i", str(rank)],
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
        return 0
