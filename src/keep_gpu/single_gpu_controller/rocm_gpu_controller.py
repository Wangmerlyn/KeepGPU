import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


class RocmGPUController(BaseGPUController):
    """
    Keep a single ROCm GPU busy by running lightweight elementwise ops
    in a background thread. Requires a ROCm-enabled torch build.
    """

    def __init__(
        self,
        *,
        rank: int,
        interval: float = 1.0,
        vram_to_keep: str | int = "1000 MB",
        busy_threshold: int = 10,
        iterations: int = 5000,
        max_allocation_retries: Optional[int] = None,
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.busy_threshold = busy_threshold
        self.iterations = iterations
        self.max_allocation_retries = max_allocation_retries
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None

        # Lazy rocm_smi import; keep handle for reuse
        try:
            import rocm_smi  # type: ignore

            self._rocm_smi = rocm_smi
        except Exception as exc:  # pragma: no cover - env-specific
            logger.debug("rocm_smi not available: %s", exc)
            self._rocm_smi = None

    def keep(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("rank %s: keep thread already running", self.rank)
            return
        self._failure_exc = None
        self._num_elements = int(self.vram_to_keep)
        if self._num_elements <= 0:
            raise ValueError("vram_to_keep must be positive")
        if self._rocm_smi:
            try:
                self._rocm_smi.rsmi_init()
            except Exception as exc:  # pragma: no cover - env-specific
                logger.debug("rsmi_init failed: %s", exc)

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._keep_loop,
            name=f"gpu-keeper-rocm-{self.rank}",
            daemon=True,
        )
        self._thread.start()
        logger.info("rank %s: ROCm keep thread started", self.rank)

    def release(self) -> None:
        if self._thread and self._thread.is_alive():
            self._stop_evt.set()
            self._thread.join()
            torch.cuda.empty_cache()
        else:
            logger.warning("rank %s: keep thread not running", self.rank)
        if self._rocm_smi:
            try:
                self._rocm_smi.rsmi_shut_down()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("rsmi_shut_down failed: %s", exc)
        logger.info("rank %s: keep thread stopped & cache cleared", self.rank)

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def _query_utilization(self) -> Optional[int]:
        if not self._rocm_smi:
            return None
        try:
            util = self._rocm_smi.rsmi_dev_busy_percent_get(self.rank)
            return int(util)
        except Exception as exc:  # pragma: no cover - env-specific
            logger.debug("ROCm utilization query failed: %s", exc)
            return None

    def _keep_loop(self) -> None:
        torch.cuda.set_device(self.rank)
        tensor = None
        attempts = 0
        num_elements = self._num_elements if self._num_elements is not None else 0
        if num_elements <= 0:
            logger.error(
                "rank %s: invalid vram_to_keep=%s", self.rank, self.vram_to_keep
            )
            return
        while not self._stop_evt.is_set():
            try:
                tensor = torch.rand(
                    num_elements,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                break
            except RuntimeError as exc:
                attempts += 1
                logger.error(
                    "rank %s: failed to allocate tensor (attempt %d%s): %s",
                    self.rank,
                    attempts,
                    (
                        f"/{self.max_allocation_retries}"
                        if self.max_allocation_retries is not None
                        else ""
                    ),
                    exc,
                )
                if (
                    self.max_allocation_retries is not None
                    and attempts >= self.max_allocation_retries
                ):
                    self._failure_exc = RuntimeError(
                        f"rank {self.rank}: failed to allocate tensor after {attempts} attempts"
                    )
                    logger.error("%s", self._failure_exc)
                    return
                time.sleep(self.interval)

        while not self._stop_evt.is_set():
            try:
                util = self._query_utilization()
                if util is not None and util > self.busy_threshold:
                    logger.debug("rank %s: GPU busy (%d%%), sleeping", self.rank, util)
                else:
                    self._run_batch(tensor)
                time.sleep(self.interval)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                time.sleep(self.interval)
            except Exception:
                logger.exception("rank %s: unexpected error", self.rank)
                time.sleep(self.interval)

    def allocation_status(self) -> Optional[Exception]:
        """
        Return allocation failure captured in the worker thread, if any.

        The reference assignment/read is thread-safe for CPython's GIL model.
        """
        return self._failure_exc

    @torch.no_grad()
    def _run_batch(self, tensor: torch.Tensor) -> None:
        tic = time.time()
        for _ in range(self.iterations):
            torch.relu_(tensor)
            if self._stop_evt.is_set():
                break
        torch.cuda.synchronize()
        toc = time.time()
        logger.debug(
            "rank %s: elementwise batch done - avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / max(1, self.iterations),
        )
