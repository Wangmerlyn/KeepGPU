"""Console script for keep_gpu."""

import os
import time
from typing import Optional, Tuple

import torch
import typer
from rich.console import Console

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities.logger import setup_logger

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
console = Console()
logger = setup_logger(__name__)


def _apply_legacy_threshold(
    vram_value: str, legacy_threshold: Optional[str], busy_threshold: int
) -> Tuple[str, int, Optional[str]]:
    """
    Interpret the deprecated --threshold flag:
    - If it parses as int, treat it as busy-threshold override.
    - Otherwise treat it as a VRAM override.
    Returns (vram, busy_threshold, mode) where mode is 'busy', 'vram', or None.
    """
    if legacy_threshold is None:
        return vram_value, busy_threshold, None

    try:
        parsed_threshold = int(legacy_threshold)
    except ValueError:
        return legacy_threshold, busy_threshold, "vram"
    else:
        return vram_value, parsed_threshold, "busy"


@app.command()
def main(
    interval: int = typer.Option(
        300, help="Interval in seconds between GPU usage checks"
    ),
    gpu_ids: Optional[str] = typer.Option(
        None,
        help="Comma-separated list of GPU IDs to monitor and benchmark on (default: all)",
    ),
    vram: str = typer.Option(
        "1GiB",
        "--vram",
        help=(
            "Amount of VRAM to keep occupied (e.g., '500MB', '1GiB', or integer in bytes). "
            "Legacy flag '--threshold' remains supported as an alias."
        ),
    ),
    legacy_threshold: Optional[str] = typer.Option(
        None,
        "--threshold",
        hidden=True,
        help="Deprecated alias. If numeric, overrides --busy-threshold; otherwise overrides --vram.",
    ),
    busy_threshold: int = typer.Option(
        -1,
        "--busy-threshold",
        "--util-threshold",
        help="Max GPU utilization threshold to trigger keeping GPU awake",
    ),
):
    """
    Keep specified GPUs awake by allocating VRAM and monitoring usage.
    """
    vram, busy_threshold, legacy_mode = _apply_legacy_threshold(
        vram, legacy_threshold, busy_threshold
    )
    if legacy_mode == "vram":
        console.print(
            "[yellow]`--threshold` for VRAM is deprecated; please use `--vram` going forward.[/yellow]"
        )
    elif legacy_mode == "busy":
        console.print(
            "[yellow]`--threshold` for utilization is deprecated; please use `--busy-threshold`.[/yellow]"
        )

    # Process GPU IDs
    if gpu_ids:
        try:
            gpu_id_list = [int(i.strip()) for i in gpu_ids.split(",")]
        except ValueError:
            console.print(
                f"[bold red]Error: Invalid characters in --gpu-ids '{gpu_ids}'. Please use comma-separated integers.[/bold red]"
            )
            raise typer.Exit(code=1)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_id_list))
        logger.info(f"Using specified GPUs: {gpu_id_list}")
        gpu_count = len(gpu_id_list)
    else:
        gpu_id_list = None
        gpu_count = torch.cuda.device_count()
        logger.info("Using all available GPUs")

    # Log settings
    logger.info(f"GPU count: {gpu_count}")
    logger.info(f"VRAM to keep occupied: {vram}")
    logger.info(f"Check interval: {interval} seconds")
    logger.info(f"Busy threshold: {busy_threshold}%")

    # Create and start Global GPU Controller
    global_controller = GlobalGPUController(
        gpu_ids=gpu_id_list,
        interval=interval,
        vram_to_keep=vram,
        busy_threshold=busy_threshold,
    )

    with global_controller:
        logger.info("Keeping GPUs awake. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Interruption received. Releasing GPUs...")


if __name__ == "__main__":
    app()
