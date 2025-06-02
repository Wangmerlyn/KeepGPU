import torch
import time
import subprocess
import re
import random
from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    gpus: int
    interval: int


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
        print(f"rank {rank}: couldn't match any, check GPU status!")
    return util


def keep(rank, args):
    torch.cuda.set_device(rank)
    print(f"rank {rank}: benchmarking {args.gpus} gpus...")
    while True:
        n = random.randint(5, 9)
        a = torch.rand((8192 * n, 8192)).cuda()
        b = torch.rand((8192 * n, 8192)).cuda()

        tic = time.time()
        for _ in range(5000):
            _ = a * b
        torch.cuda.synchronize()
        toc = time.time()
        if rank == 0:
            print(f"benchmark 5K matmul: time span: {(toc - tic) * 1000 / 5000:.2f}ms")

        time.sleep(args.interval)
        while get_gpu_util(rank) > 10:
            print(f"rank {rank}: GPU busy, sleeping...")
            time.sleep(args.interval)
        print(f"rank {rank} resumes")


def run_benchmark(gpus=1, interval=100):
    args = BenchmarkConfig(gpus=gpus, interval=interval)
    torch.multiprocessing.spawn(keep, args=(args,), nprocs=gpus, join=True)
