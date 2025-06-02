import subprocess
import time
import torch

from .benchmark import run_benchmark


def check_gpu_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    gpu_usages = [int(line.split()[0]) for line in result.stdout.strip().split("\n")]
    return any(usage > 0 for usage in gpu_usages)


def run():
    idle_count = 0
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    while True:
        if not check_gpu_usage():
            idle_count += 1
        else:
            idle_count = 0

        if idle_count >= 1:
            run_benchmark(gpu_count)
            idle_count = 0

        time.sleep(300)
