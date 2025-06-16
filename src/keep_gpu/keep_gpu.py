import argparse
import os
import subprocess
import time
import torch

from .benchmark import run_benchmark
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
    return parser.parse_args()


def check_gpu_usage(gpu_ids=None):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    usage_lines = result.stdout.strip().split("\n")
    usages = [int(line.strip()) for line in usage_lines]

    if gpu_ids is not None:
        usages = [usages[i] for i in gpu_ids if i < len(usages)]

    return any(usage > 0 for usage in usages)


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

    idle_count = 0
    logger.info(f"GPU count: {gpu_count}")
    while True:
        if not check_gpu_usage(gpu_ids):
            idle_count += 1
        else:
            idle_count = 0

        if idle_count >= 1:
            run_benchmark(gpu_count)
            idle_count = 0

        time.sleep(args.interval)
