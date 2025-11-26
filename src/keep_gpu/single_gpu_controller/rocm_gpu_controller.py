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
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self.busy_threshold = busy_threshold
        self.iterations = iterations
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None

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
        if not (self._thread and self._thread.is_alive()):
            logger.warning("rank %s: keep thread not running", self.rank)
            return
        self._stop_evt.set()
        self._thread.join()
        torch.cuda.empty_cache()
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
        while not self._stop_evt.is_set():
            try:
                tensor = torch.rand(
                    self.vram_to_keep,
                    device=self.device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                break
            except RuntimeError as exc:
                logger.error("rank %s: failed to allocate tensor: %s", self.rank, exc)
                time.sleep(self.interval)
        if tensor is None:
            logger.error("rank %s: failed to allocate tensor, exiting loop", self.rank)
            raise RuntimeError("Failed to allocate tensor for ROCm GPU keeping")

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
            "rank %s: elementwise batch done – avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / max(1, self.iterations),
        )
