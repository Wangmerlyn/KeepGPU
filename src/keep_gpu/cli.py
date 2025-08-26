"""Console script for keep_gpu."""

import os
import time
from typing import Optional

import torch
import typer
from rich.console import Console

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities.logger import setup_logger

app = typer.Typer()
console = Console()
logger = setup_logger(__name__)


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
        help="Amount of VRAM to keep occupied (e.g., '500MB', '1GiB', or integer in bytes)",
    ),
    threshold: int = typer.Option(
        -1,
        help="Max GPU utilization threshold to trigger keeping GPU awake",
    ),
):
    """
    Keep specified GPUs awake by allocating VRAM and monitoring usage.
    """
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
    logger.info(f"Busy threshold: {threshold}%")

    # Create and start Global GPU Controller
    global_controller = GlobalGPUController(
        gpu_ids=gpu_id_list,
        interval=interval,
        vram_to_keep=vram,
        busy_threshold=threshold,
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
