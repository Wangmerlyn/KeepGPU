import threading
from typing import List, Optional, Union

from keep_gpu.utilities.humanized_input import parse_size, parse_vram_to_elements
from keep_gpu.utilities.logger import setup_logger
from keep_gpu.utilities.session_config import (
    DEFAULT_BUSY_THRESHOLD,
    VisibleRankValidationError,
    validate_busy_threshold,
    validate_gpu_ids,
    validate_interval,
)
from keep_gpu.utilities.platform_manager import (
    ComputingPlatform,
    DeviceEnumerationUnavailableError,
    get_platform,
    visible_torch_device_count,
)

logger = setup_logger(__name__)


class ControllerStartupUnavailable(Exception):
    """Expected hardware/platform unavailability during controller startup."""


class NoGPUAvailableError(ControllerStartupUnavailable, ValueError):
    """No visible GPU devices are available for a global keep session."""


class UnsupportedControllerPlatformError(
    ControllerStartupUnavailable, NotImplementedError
):
    """The current platform cannot run the global GPU controller."""


class InvalidVisibleGPUSelectionError(ValueError):
    """Public gpu_ids selection is not valid for the visible device set."""


def _resolve_visible_gpu_ids(gpu_ids: Optional[List[int]]) -> List[int]:
    try:
        visible_count = visible_torch_device_count()
    except DeviceEnumerationUnavailableError as exc:
        raise NoGPUAvailableError(str(exc)) from exc
    if gpu_ids is None:
        return list(range(visible_count))
    invalid_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= visible_count]
    if invalid_ids:
        raise InvalidVisibleGPUSelectionError(
            "gpu_ids must be visible device ordinals less than "
            f"{visible_count}; got {invalid_ids}"
        )
    return gpu_ids


class GlobalGPUController:
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        interval: Union[int, float] = 300,
        vram_to_keep: Union[int, str] = "1GiB",
        busy_threshold: int = DEFAULT_BUSY_THRESHOLD,
    ):
        self.interval = validate_interval(interval)
        self.busy_threshold = validate_busy_threshold(busy_threshold)
        parse_vram_to_elements(vram_to_keep)
        self.vram_to_keep = vram_to_keep
        gpu_ids = validate_gpu_ids(gpu_ids)
        self.computing_platform = get_platform()
        if self.computing_platform == ComputingPlatform.CUDA:
            from keep_gpu.single_gpu_controller.cuda_gpu_controller import (
                CudaGPUController,
            )

            controller_cls = CudaGPUController
        elif self.computing_platform == ComputingPlatform.ROCM:
            from keep_gpu.single_gpu_controller.rocm_gpu_controller import (
                RocmGPUController,
            )

            controller_cls = RocmGPUController
        elif self.computing_platform == ComputingPlatform.MACM:
            from keep_gpu.single_gpu_controller.macm_gpu_controller import (
                MacMGPUController,
            )

            controller_cls = MacMGPUController
        else:
            raise UnsupportedControllerPlatformError(
                f"GlobalGPUController not implemented for platform {self.computing_platform}"
            )

        if self.computing_platform == ComputingPlatform.MACM:
            if gpu_ids is None:
                self.gpu_ids = [0]
            elif gpu_ids == [0]:
                self.gpu_ids = gpu_ids
            else:
                raise InvalidVisibleGPUSelectionError(
                    f"MACM platform only supports gpu_ids=[0] or None, got {gpu_ids}"
                )
        elif self.computing_platform in (
            ComputingPlatform.CUDA,
            ComputingPlatform.ROCM,
        ):
            self.gpu_ids = _resolve_visible_gpu_ids(gpu_ids)
        else:
            self.gpu_ids = gpu_ids

        if not self.gpu_ids:
            raise NoGPUAvailableError("No GPUs available for GlobalGPUController")

        try:
            self.controllers = [
                controller_cls(
                    rank=i,
                    interval=self.interval,
                    vram_to_keep=self.vram_to_keep,
                    busy_threshold=self.busy_threshold,
                )
                for i in self.gpu_ids
            ]
        except VisibleRankValidationError as exc:
            raise NoGPUAvailableError(str(exc)) from exc
        except DeviceEnumerationUnavailableError as exc:
            raise NoGPUAvailableError(str(exc)) from exc

    def keep(self) -> None:
        started = []
        for ctrl in self.controllers:
            try:
                ctrl.keep()
            except Exception:
                for started_ctrl in reversed(started):
                    try:
                        started_ctrl.release()
                    except Exception as cleanup_exc:
                        logger.warning(
                            "Failed to roll back controller rank %s after start failure: %s",
                            getattr(started_ctrl, "rank", "unknown"),
                            cleanup_exc,
                        )
                raise
            started.append(ctrl)

    @staticmethod
    def parse_size(text: str) -> int:
        return parse_size(text)

    def release(self) -> None:
        threads = []
        errors = []
        errors_lock = threading.Lock()

        def _release_controller(ctrl) -> None:
            try:
                ctrl.release()
            except Exception as exc:
                with errors_lock:
                    errors.append((getattr(ctrl, "rank", "unknown"), exc))

        for ctrl in self.controllers:
            t = threading.Thread(target=_release_controller, args=(ctrl,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        if errors:
            details = "; ".join(
                (
                    str(exc)
                    if str(exc).startswith(f"rank {rank}:")
                    else f"rank {rank}: {exc}"
                )
                for rank, exc in errors
            )
            raise RuntimeError(f"Failed to release GPU controllers: {details}")

    def runtime_error(self) -> Optional[Exception]:
        """Return the first terminal child-controller runtime error, if any."""
        for ctrl in self.controllers:
            allocation_status = getattr(ctrl, "allocation_status", None)
            if not callable(allocation_status):
                continue
            try:
                error = allocation_status()
            except Exception as exc:  # noqa: BLE001 - health hook failure is health
                error = exc
            if error is None:
                continue
            rank = getattr(ctrl, "rank", "unknown")
            message = str(error)
            if message.startswith(f"rank {rank}:"):
                return error
            return RuntimeError(f"rank {rank}: {message}")
        return None

    def __enter__(self) -> "GlobalGPUController":
        self.keep()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
