import torch
import time
import subprocess
import re
import random
from dataclasses import dataclass
from torch.multiprocessing.spawn import spawn

from keep_gpu.utilities.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BenchmarkConfig:
    gpus: int
    interval: int
    matmul_iterations: int = 5000  # number of matmul calculations per loop


def get_gpu_util(rank):
    cmds = ["nvidia-smi", "-i", str(rank)]
    proc = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    outputs = stdout.decode("utf-8").split("\n")

    util = 0
    for output in outputs[::-1]:
        if "Default" in output:
            util = int(re.findall(r"\d+", output)[-1])
            break
    else:
        logger.warning(f"rank {rank}: couldn't match any, check GPU status!")
    return util


def keep(rank, args):
    torch.cuda.set_device(rank)
    logger.info(f"rank {rank}: benchmarking {args.gpus} gpus...")

    while True:
        try:
            n = random.randint(5, 9)
            a = torch.rand((8192 * n, 4096), device="cuda")
            b = torch.rand((4096, 8192 * 5), device="cuda")

            tic = time.time()
            for _ in range(args.matmul_iterations):
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
            toc = time.time()

            logger.info(
                f"benchmark {rank} matmul: time span: {(toc - tic) * 1000 / 5000:.2f}ms"
            )

            time.sleep(args.interval)

            while get_gpu_util(rank) > 10:
                logger.warning(f"rank {rank}: GPU busy, sleeping...")
                time.sleep(args.interval)

            logger.info(f"rank {rank} resumes")

        except RuntimeError as e:
            logger.error(f"rank {rank}: RuntimeError encountered: {e}")
            if "out of memory" in str(e).lower():
                logger.warning(
                    f"rank {rank}: CUDA OOM — clearing cache and sleeping..."
                )
                torch.cuda.empty_cache()
            time.sleep(args.interval)

        except KeyboardInterrupt:
            logger.info(f"rank {rank}: Interrupted by user. Exiting keep loop.")
            break

        except Exception as e:
            logger.exception(f"rank {rank}: Unexpected error: {e}")
            time.sleep(args.interval)


def run_benchmark(gpus=1, interval=100):
    args = BenchmarkConfig(gpus=gpus, interval=interval)
    spawn(keep, args=(args,), nprocs=gpus, join=True)
