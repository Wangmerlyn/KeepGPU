from __future__ import annotations

import threading
import time
from typing import Optional

import torch

from keep_gpu.single_gpu_controller.base_gpu_controller import BaseGPUController
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.platform_manager import visible_torch_device_count
from keep_gpu.utilities.rocm_visibility import resolve_rocm_visible_rank_to_smi_index
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    normalize_utilization_percent,
    validate_busy_threshold,
    validate_positive_integer,
    validate_rank_type,
    validate_visible_rank,
)

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
        vram_to_keep: str | int = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
        iterations: int = 5000,
        max_allocation_retries: Optional[int] = None,
    ):
        super().__init__(vram_to_keep=vram_to_keep, interval=interval)
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        self.iterations = validate_positive_integer(iterations, "iterations")
        self.max_allocation_retries = (
            None
            if max_allocation_retries is None
            else validate_positive_integer(
                max_allocation_retries, "max_allocation_retries"
            )
        )
        rank = validate_rank_type(rank)
        rank = validate_visible_rank(rank, visible_torch_device_count())
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")
        self._stop_evt: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._failure_exc: Optional[Exception] = None
        self._num_elements: Optional[int] = None
        self._rocm_smi_initialized = False

        # Lazy rocm_smi import; keep handle for reuse
        try:
            import rocm_smi  # type: ignore

            self._rocm_smi = rocm_smi
        except Exception as exc:  # pragma: no cover - env-specific
            logger.debug("rocm_smi not available: %s", exc)
            self._rocm_smi = None

    def keep(self) -> None:
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
        self._ensure_rocm_smi_initialized()

        self._stop_evt = threading.Event()
        startup_evt = threading.Event()
        startup_errors: list[Exception] = []
        self._thread = threading.Thread(
            target=self._keep_loop,
            args=(startup_evt, startup_errors),
            name=f"gpu-keeper-rocm-{self.rank}",
            daemon=True,
        )
        self._thread.start()
        startup_timeout = 5.0
        if not startup_evt.wait(startup_timeout):
            stop_evt = self._stop_evt
            if stop_evt is not None:
                stop_evt.set()
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                self._shutdown_rocm_smi()
                raise RuntimeError(
                    f"rank {self.rank}: ROCm keep thread did not complete startup "
                    f"within {startup_timeout:.1f}s"
                )
            self._thread = None
            self._stop_evt = None
            self._shutdown_rocm_smi()
            raise RuntimeError(
                f"rank {self.rank}: ROCm keep thread exited before startup completed"
            )
        if startup_errors:
            self._thread.join(timeout=1.0)
            self._thread = None
            self._stop_evt = None
            self._shutdown_rocm_smi()
            raise startup_errors[0]
        logger.info("rank %s: ROCm keep thread started", self.rank)

    def _shutdown_rocm_smi(self) -> None:
        if not self._rocm_smi:
            return
        try:
            self._rocm_smi.rsmi_shut_down()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - best effort
            logger.debug("rsmi_shut_down failed: %s", exc)
        finally:
            self._rocm_smi_initialized = False

    def _ensure_rocm_smi_initialized(self) -> bool:
        if not self._rocm_smi:
            return False
        if self._rocm_smi_initialized:
            return True
        try:
            self._rocm_smi.rsmi_init()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - env-specific
            logger.debug("rsmi_init failed: %s", exc)
            return False
        self._rocm_smi_initialized = True
        return True

    def release(self) -> None:
        try:
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

            if thread and thread.is_alive():
                stop_evt = self._stop_evt
                if stop_evt is None:
                    raise RuntimeError(f"rank {self.rank}: stop event missing")
                assert stop_evt is not None
                stop_evt.set()
                join_timeout = max(2.0, min(float(self.interval) + 2.0, 30.0))
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "rank %s: ROCm keep thread did not stop within %.1fs",
                        self.rank,
                        join_timeout,
                    )
                    raise TimeoutError(
                        f"rank {self.rank}: ROCm keep thread did not stop within {join_timeout:.1f}s"
                    )
                torch.cuda.empty_cache()
                self._thread = None
                self._stop_evt = None
            else:
                logger.warning("rank %s: keep thread not running", self.rank)
                return
        finally:
            self._shutdown_rocm_smi()
        logger.info("rank %s: keep thread stopped & cache cleared", self.rank)

    def __enter__(self):
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def _query_utilization(self) -> Optional[int]:
        for attempt in range(2):
            if not self._ensure_rocm_smi_initialized():
                return None
            try:
                smi_index = resolve_rocm_visible_rank_to_smi_index(
                    self.rank,
                    self._rocm_smi,
                )
                if smi_index is None:
                    return None
                util = self._rocm_smi.rsmi_dev_busy_percent_get(smi_index)
                utilization = normalize_utilization_percent(util)
                return int(utilization) if utilization is not None else None
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - env-specific
                logger.debug("ROCm utilization query failed: %s", exc)
                if attempt == 0:
                    self._rocm_smi_initialized = False
                    continue
                return None
        return None

    def _keep_loop(
        self,
        startup_evt: Optional[threading.Event] = None,
        startup_errors: Optional[list[Exception]] = None,
    ) -> None:
        startup_confirmed = startup_evt is None

        def confirm_startup() -> None:
            nonlocal startup_confirmed
            if startup_confirmed:
                return
            startup_confirmed = True
            assert startup_evt is not None
            startup_evt.set()

        def record_worker_failure(exc: Exception) -> None:
            failure = RuntimeError(
                f"rank {self.rank}: unexpected ROCm keep worker failure: {exc}"
            )
            if not startup_confirmed and startup_errors is not None:
                startup_errors.append(failure)
            else:
                self._failure_exc = failure
            confirm_startup()
            logger.exception("%s", failure)

        stop_evt = self._stop_evt
        if stop_evt is None:
            exc = RuntimeError(f"rank {self.rank}: stop event not initialized")
            logger.error("%s", exc)
            if startup_errors is not None:
                startup_errors.append(exc)
            if startup_evt is not None:
                startup_evt.set()
            return
        assert stop_evt is not None

        try:
            torch.cuda.set_device(self.rank)
        except Exception as exc:  # noqa: BLE001 - surface backend startup failure
            logger.error("rank %s: ROCm startup failed: %s", self.rank, exc)
            if startup_errors is not None:
                startup_errors.append(exc)
            if startup_evt is not None:
                startup_evt.set()
            return
        tensor = None
        attempts = 0
        num_elements = self._num_elements if self._num_elements is not None else 0
        if num_elements <= 0:
            exc = RuntimeError(
                f"rank {self.rank}: invalid vram_to_keep={self.vram_to_keep}"
            )
            logger.error("%s", exc)
            if startup_errors is not None:
                startup_errors.append(exc)
            if startup_evt is not None:
                startup_evt.set()
            return
        while not stop_evt.is_set():
            try:
                util = self._query_utilization()
                if not self._should_run_batch(util, self.busy_threshold):
                    confirm_startup()
                    logger.debug(
                        "rank %s: GPU utilization unavailable or busy (%s), deferring allocation",
                        self.rank,
                        "n/a" if util is None else f"{util}%",
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
                confirm_startup()
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
                if "out of memory" not in str(exc).lower():
                    record_worker_failure(exc)
                    return
                if (
                    self.max_allocation_retries is not None
                    and attempts >= self.max_allocation_retries
                ):
                    failure = RuntimeError(
                        f"rank {self.rank}: failed to allocate tensor after {attempts} attempts"
                    )
                    if not startup_confirmed and startup_errors is not None:
                        startup_errors.append(failure)
                    else:
                        self._failure_exc = failure
                    confirm_startup()
                    logger.error("%s", failure)
                    return
                torch.cuda.empty_cache()
                confirm_startup()
                if stop_evt.wait(self.interval):
                    return
            except Exception as exc:
                record_worker_failure(exc)
                return

        if tensor is None:
            logger.error("rank %s: failed to allocate tensor, exiting loop", self.rank)
            return
        assert tensor is not None

        while not stop_evt.is_set():
            try:
                util = self._query_utilization()
                if self._should_run_batch(util, self.busy_threshold):
                    self._run_batch(tensor)
                else:
                    logger.debug(
                        "rank %s: GPU utilization unavailable or busy (%s), sleeping",
                        self.rank,
                        "n/a" if util is None else f"{util}%",
                    )
                if stop_evt.wait(self.interval):
                    break
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    if stop_evt.wait(self.interval):
                        break
                    continue
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected ROCm keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return
            except Exception as exc:
                self._failure_exc = RuntimeError(
                    f"rank {self.rank}: unexpected ROCm keep worker failure: {exc}"
                )
                logger.exception("%s", self._failure_exc)
                return

    def allocation_status(self) -> Optional[Exception]:
        """
        Return allocation failure captured in the worker thread, if any.

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
        torch.cuda.synchronize()
        toc = time.time()
        logger.debug(
            "rank %s: elementwise batch done - avg %.2f ms",
            self.rank,
            (toc - tic) * 1000 / max(1, self.iterations),
        )
