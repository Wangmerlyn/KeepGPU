import gc
import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import ComputingPlatform

logger = setup_logger(__name__)


class MacMGPUController(BaseGPUController):
    def __init__(
        self,
        *,
        rank: int = 0,
        interval: float = 1.0,
        vram_to_keep: str | int = "1000 MB",
        busy_threshold: int = 10,
        iterations: int = 5000,
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        if rank != 0:
            raise ValueError("MPS only supports device 0; rank must be 0")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS backend is not available")

        self.rank = rank
        self.device = torch.device("mps")
        self.busy_threshold = busy_threshold
        self.iterations = iterations
        self.platform = ComputingPlatform.MACM

        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._num_elements: Optional[int] = None

        logger.debug(
            "rank %s: busy_threshold=%s ignored on macOS MPS (API compatibility)",
            self.rank,
            self.busy_threshold,
        )

    def keep(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("rank %s: keep thread already running", self.rank)
            return

        self._num_elements = int(self.vram_to_keep)
        if self._num_elements <= 0:
            raise ValueError("vram_to_keep must be positive")

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._keep_loop,
            name=f"gpu-keeper-macm-{self.rank}",
            daemon=True,
        )
        self._thread.start()
        logger.info("rank %s: MPS keep thread started", self.rank)

    def release(self) -> None:
        if not (self._thread and self._thread.is_alive()):
            logger.warning("rank %s: keep thread not running", self.rank)
            return

        stop_evt = self._stop_evt
        if stop_evt is None:
            logger.warning("rank %s: stop event missing; skipping release", self.rank)
            return

        stop_evt.set()
        join_timeout = max(2.0, min(float(self.interval) + 2.0, 30.0))
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            logger.warning(
                "rank %s: MPS keep thread did not stop within %.1fs",
                self.rank,
                join_timeout,
            )
            return

        torch.mps.empty_cache()
        gc.collect()
        logger.info("rank %s: keep thread stopped & cache cleared", self.rank)

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def _keep_loop(self) -> None:
        stop_evt = self._stop_evt
        if stop_evt is None:
            logger.error("rank %s: stop event not initialized", self.rank)
            return

        num_elements = self._num_elements if self._num_elements is not None else 0
        if num_elements <= 0:
            logger.error(
                "rank %s: invalid vram_to_keep=%s", self.rank, self.vram_to_keep
            )
            return

        tensor = None
        while not stop_evt.is_set():
            try:
                tensor = torch.rand(
                    num_elements,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                break
            except RuntimeError as exc:
                logger.error("rank %s: failed to allocate tensor: %s", self.rank, exc)
                torch.mps.empty_cache()
                gc.collect()
                if stop_evt.wait(self.interval):
                    return

        if tensor is None:
            logger.error("rank %s: failed to allocate tensor, exiting loop", self.rank)
            return

        while not stop_evt.is_set():
            try:
                self._run_batch(tensor)
                if stop_evt.wait(self.interval):
                    break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.mps.empty_cache()
                    gc.collect()
                if stop_evt.wait(self.interval):
                    break
            except Exception:
                logger.exception("rank %s: unexpected error", self.rank)
                if stop_evt.wait(self.interval):
                    break

    @torch.no_grad()
    def _run_batch(self, tensor: torch.Tensor) -> None:
        stop_evt = self._stop_evt

        tic = time.time()
        for _ in range(self.iterations):
            torch.relu_(tensor)
            if stop_evt is not None and stop_evt.is_set():
                break
        torch.mps.synchronize()
        toc = time.time()

        logger.debug(
            "rank %s: elementwise batch done - avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / max(1, self.iterations),
        )
