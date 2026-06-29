from __future__ import annotations

import gc
import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import ComputingPlatform
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    validate_busy_threshold,
    validate_positive_integer,
)

logger = setup_logger(__name__)


class MacMGPUController(BaseGPUController):
    def __init__(
        self,
        *,
        rank: int = 0,
        interval: float = 1.0,
        vram_to_keep: str | int = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
        iterations: int = 5000,
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        if rank != 0:
            raise ValueError("MPS only supports device 0; rank must be 0")
        iterations = validate_positive_integer(iterations, "iterations")
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS backend is not available")

        self.rank = rank
        self.device = torch.device("mps")
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        self.iterations = iterations
        self.platform = ComputingPlatform.MACM

        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None

    def keep(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("rank %s: keep thread already running", self.rank)
            return

        self._failure_exc = None
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
        thread = self._thread
        if thread is not None and not thread.is_alive():
            stop_evt = self._stop_evt
            if (stop_evt is not None and stop_evt.is_set()) or getattr(
                self, "_failure_exc", None
            ) is not None:
                torch.mps.empty_cache()
                gc.collect()
                self._thread = None
                self._stop_evt = None
                logger.info(
                    "rank %s: previously stopping keep thread exited; cache cleared",
                    self.rank,
                )
                return

        if not (thread and thread.is_alive()):
            logger.warning("rank %s: keep thread not running", self.rank)
            return

        stop_evt = self._stop_evt
        if stop_evt is None:
            raise RuntimeError(f"rank {self.rank}: stop event missing")

        stop_evt.set()
        join_timeout = max(2.0, min(float(self.interval) + 2.0, 30.0))
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            logger.warning(
                "rank %s: MPS keep thread did not stop within %.1fs",
                self.rank,
                join_timeout,
            )
            raise TimeoutError(
                f"rank {self.rank}: MPS keep thread did not stop within {join_timeout:.1f}s"
            )

        torch.mps.empty_cache()
        gc.collect()
        self._thread = None
        self._stop_evt = None
        logger.info("rank %s: keep thread stopped & cache cleared", self.rank)

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def _keep_loop(self) -> None:
        stop_evt = self._stop_evt
        if stop_evt is None:
            self._failure_exc = RuntimeError(
                f"rank {self.rank}: stop event not initialized"
            )
            logger.error("%s", self._failure_exc)
            return

        num_elements = self._num_elements if self._num_elements is not None else 0
        if num_elements <= 0:
            self._failure_exc = RuntimeError(
                f"rank {self.rank}: invalid vram_to_keep={self.vram_to_keep}"
            )
            logger.error("%s", self._failure_exc)
            return

        tensor = None
        while not stop_evt.is_set():
            try:
                if not self._should_run_batch(None, self.busy_threshold):
                    logger.debug(
                        "rank %s: MPS utilization unavailable; deferring allocation because busy_threshold=%s",
                        self.rank,
                        self.busy_threshold,
                    )
                    if stop_evt.wait(self.interval):
                        return
                    continue
                tensor = torch.rand(
                    num_elements,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                break
            except RuntimeError as exc:
                logger.error("rank %s: failed to allocate tensor: %s", self.rank, exc)
                if "out of memory" in str(exc).lower():
                    torch.mps.empty_cache()
                    gc.collect()
                    if stop_evt.wait(self.interval):
                        return
                    continue
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected MPS keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected MPS keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return

        if tensor is None:
            logger.error("rank %s: failed to allocate tensor, exiting loop", self.rank)
            return

        while not stop_evt.is_set():
            try:
                if self._should_run_batch(None, self.busy_threshold):
                    self._run_batch(tensor)
                else:
                    logger.debug(
                        "rank %s: MPS utilization unavailable; sleeping because busy_threshold=%s",
                        self.rank,
                        self.busy_threshold,
                    )
                if stop_evt.wait(self.interval):
                    break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.mps.empty_cache()
                    gc.collect()
                    if stop_evt.wait(self.interval):
                        break
                    continue
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected MPS keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected MPS keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return

    def allocation_status(self) -> Optional[Exception]:
        """
        Return fatal worker failure captured after startup, if any.

        The reference assignment/read is thread-safe for CPython's GIL model.
        """
        return self._failure_exc

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
