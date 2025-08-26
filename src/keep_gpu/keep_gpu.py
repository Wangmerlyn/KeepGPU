import argparse
import os
import time

import torch

from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController
from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GPU Idle Monitor and Benchmark Trigger"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds between GPU usage checks",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to monitor and benchmark on (default: all)",
    )
    parser.add_argument(
        "--vram",
        type=str,
        default="1GiB",
        help="Amount of VRAM to keep occupied (e.g., '500MB', '1GiB', or integer in bytes)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=-1,
        help="Max gpu utilization threshold to trigger keeping GPU awake",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    if args.gpu_ids:
        gpu_ids = [int(i.strip()) for i in args.gpu_ids.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        logger.info(f"Using specified GPUs: {gpu_ids}")
        gpu_count = len(gpu_ids)
    else:
        gpu_ids = None
        gpu_count = torch.cuda.device_count()
        logger.info("Using all available GPUs")

    logger.info(f"GPU count: {gpu_count}")
    logger.info(f"VRAM to keep occupied: {args.vram}")
    logger.info(f"Check interval: {args.interval} seconds")
    logger.info(f"Busy threshold {args.threshold}%")
    global_controller = GlobalGPUController(
        gpu_ids=gpu_ids,
        interval=args.interval,
        vram_to_keep=args.vram,
        busy_threshold=args.threshold,
    )
    global_controller.keep()
    while True:
        time.sleep(1)
