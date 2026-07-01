from __future__ import annotations

import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.gpu_monitor import get_gpu_utilization
from keep_gpu.utilities.humanized_input import parse_size
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import (
    ComputingPlatform,
    visible_torch_device_count,
)
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    validate_busy_threshold,
    validate_positive_integer,
    validate_rank_type,
    validate_visible_rank,
)

logger = setup_logger(__name__)


class CudaGPUController(BaseGPUController):
    """CudaGPUController
    Keep a single CUDA GPU busy by repeatedly running lightweight
    elementwise workloads in a background thread.

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
        relu_iterations: int = 5000,
        vram_to_keep: str | int = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
    ):
        """
        Args:
            rank (int): Local CUDA device index to occupy.
            interval (float, optional): Sleep time (seconds) between workload
                batches. Defaults to 1.0.
            relu_iterations (int, optional): Number of in-place ReLU ops per
                batch.
            vram_to_keep (int or str, optional): Amount of VRAM to keep busy,
                e.g. `"1GiB"`, `"20 GB"`, or an integer like `1000 * 1000`.
                This represents the total size of the matrix allocated to
                occupy the GPU.
            busy_threshold (int, optional): Defaults to 25. If current
                utilization (%) exceeds this threshold, or utilization is
                unavailable, the worker inserts extra sleeps to avoid hogging
                the GPU. Use -1 only to disable utilization backoff.

        """
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.relu_iterations = validate_positive_integer(
            relu_iterations, "relu_iterations"
        )
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        rank = validate_rank_type(rank)
        rank = validate_visible_rank(rank, visible_torch_device_count())
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.interval = interval
        self.platform = ComputingPlatform.CUDA

        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None

    @staticmethod
    def parse_size(text: str) -> int:
        return parse_size(text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keep(self) -> None:
        """Launch the background thread that keeps the GPU busy."""
        if self._thread and self._thread.is_alive():
            if self._stop_evt is not None and self._stop_evt.is_set():
                raise RuntimeError(
                    f"rank {self.rank}: previous keep thread startup did not complete"
                )
            logger.warning("rank %s: keep thread already running", self.rank)
            return

        self._failure_exc = None
        self._num_elements = int(self.vram_to_keep)
        if self._num_elements <= 0:
            raise ValueError("vram_to_keep must be positive")

        self._stop_evt = threading.Event()
        startup_evt = threading.Event()
        startup_errors: list[Exception] = []
        self._thread = threading.Thread(
            target=self._keep_loop,
            args=(startup_evt, startup_errors),
            name=f"gpu-keeper-{self.rank}",
            daemon=True,  # daemon so program can exit cleanly
        )
        self._thread.start()
        startup_timeout = 5.0
        if not startup_evt.wait(startup_timeout):
            stop_evt = self._stop_evt
            if stop_evt is not None:
                stop_evt.set()
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError(
                    f"rank {self.rank}: keep thread did not complete startup within "
                    f"{startup_timeout:.1f}s"
                )
            self._thread = None
            self._stop_evt = None
            raise RuntimeError(
                f"rank {self.rank}: keep thread exited before startup completed"
            )
        if startup_errors:
            self._thread.join(timeout=1.0)
            self._thread = None
            self._stop_evt = None
            raise startup_errors[0]
        logger.info("rank %s: keep thread started", self.rank)

    def release(self) -> None:
        """
        Stop the background thread and clear CUDA cache so the memory
        becomes immediately available to other code.
        """
        thread = self._thread
        if thread is not None and not thread.is_alive():
            stop_evt = self._stop_evt
            if (stop_evt is not None and stop_evt.is_set()) or getattr(
                self, "_failure_exc", None
            ) is not None:
                torch.cuda.empty_cache()
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
        assert stop_evt is not None

        stop_evt.set()
        join_timeout = max(2.0, min(float(self.interval) + 2.0, 30.0))
        thread.join(timeout=join_timeout)
        if thread.is_alive():
            logger.warning(
                "rank %s: keep thread did not stop within %.1fs",
                self.rank,
                join_timeout,
            )
            raise TimeoutError(
                f"rank {self.rank}: keep thread did not stop within {join_timeout:.1f}s"
            )
        torch.cuda.empty_cache()
        self._thread = None
        self._stop_evt = None
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
    def _keep_loop(
        self,
        startup_evt: Optional[threading.Event] = None,
        startup_errors: Optional[list[Exception]] = None,
    ) -> None:
        """Internal: run workloads until stop event is set."""
        startup_confirmed = startup_evt is None

        def confirm_startup() -> None:
            nonlocal startup_confirmed
            if startup_confirmed:
                return
            startup_confirmed = True
            assert startup_evt is not None
            startup_evt.set()

        def record_startup_failure(exc: Exception) -> None:
            if not startup_confirmed and startup_errors is not None:
                startup_errors.append(exc)
            else:
                self._failure_exc = exc
            confirm_startup()

        def record_worker_failure(exc: Exception) -> None:
            failure = RuntimeError(
                f"rank {self.rank}: unexpected CUDA keep worker failure: {exc}"
            )
            record_startup_failure(failure)
            logger.exception("%s", failure)

        stop_evt = self._stop_evt
        if stop_evt is None:
            exc = RuntimeError(f"rank {self.rank}: stop event not initialized")
            logger.error("%s", exc)
            record_startup_failure(exc)
            return
        assert stop_evt is not None

        try:
            torch.cuda.set_device(self.rank)
        except Exception as exc:  # noqa: BLE001 - surface backend startup failure
            logger.error("rank %s: CUDA startup failed: %s", self.rank, exc)
            record_startup_failure(exc)
            return
        num_elements = self._num_elements if self._num_elements is not None else 0
        if num_elements <= 0:
            exc = RuntimeError(
                f"rank {self.rank}: invalid vram_to_keep={self.vram_to_keep}"
            )
            logger.error("%s", exc)
            record_startup_failure(exc)
            return
        matrix = None
        while not stop_evt.is_set():
            try:
                gpu_utilization = self._monitor_utilization(self.rank)
                if not self._should_run_batch(gpu_utilization, self.busy_threshold):
                    confirm_startup()
                    logger.debug(
                        "rank %s: GPU utilization unavailable or busy (%s), deferring allocation",
                        self.rank,
                        "n/a" if gpu_utilization is None else f"{gpu_utilization}%",
                    )
                    if stop_evt.wait(self.interval):
                        return
                    continue
                matrix = torch.rand(
                    num_elements,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                confirm_startup()
                break
            except RuntimeError as e:
                logger.error("rank %s: failed to allocate matrix: %s", self.rank, e)
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    confirm_startup()
                    if stop_evt.wait(self.interval):
                        return
                    continue
                record_worker_failure(e)
                return
            except Exception as exc:  # noqa: BLE001 - surface backend startup failure
                record_worker_failure(exc)
                return
        if matrix is None:
            if not startup_confirmed:
                exc = RuntimeError(
                    f"rank {self.rank}: stopped before CUDA startup allocation"
                )
                logger.error("%s", exc)
                record_startup_failure(exc)
            else:
                logger.debug(
                    "rank %s: exiting before CUDA allocation after startup confirmation",
                    self.rank,
                )
            return
        while not stop_evt.is_set():
            try:
                gpu_utilization = self._monitor_utilization(self.rank)
                if not self._should_run_batch(gpu_utilization, self.busy_threshold):
                    logger.debug(
                        "rank %s: GPU utilization unavailable or busy (%s), sleeping longer",
                        self.rank,
                        "n/a" if gpu_utilization is None else f"{gpu_utilization}%",
                    )
                else:
                    self._run_relu_batch(matrix)
                if stop_evt.wait(self.interval):
                    break
            except RuntimeError as e:
                # Handle OOM by clearing cache; then sleep and continue
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if stop_evt.wait(self.interval):
                        break
                    continue
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected CUDA keep worker failure: {e}"
                )
                logger.exception("%s", self._failure_exc)
                return
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected CUDA keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return

    def allocation_status(self) -> Optional[Exception]:
        """
        Return fatal worker failure captured after startup, if any.

        The reference assignment/read is thread-safe for CPython's GIL model.
        """
        return self._failure_exc

    # ------------------------------------------------------------------
    # Workload implementation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _run_relu_batch(self, matrix: torch.Tensor) -> None:
        """Run a batch of in-place ReLU ops to keep GPU busy."""
        stop_evt = self._stop_evt

        tic = time.time()
        for _ in range(self.relu_iterations):
            torch.relu_(matrix)
            if stop_evt is not None and stop_evt.is_set():
                break
        torch.cuda.synchronize()
        toc = time.time()

        logger.debug(
            "rank %s: relu ops batch done - avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / max(1, self.relu_iterations),
        )

    # ------------------------------------------------------------------
    # Utilization monitor
    # ------------------------------------------------------------------
    @staticmethod
    def _monitor_utilization(rank: int) -> Optional[int]:
        """
        Return current GPU utilization (%) for `rank`.
        Returns None when telemetry is unavailable.
        """
        return get_gpu_utilization(rank)
