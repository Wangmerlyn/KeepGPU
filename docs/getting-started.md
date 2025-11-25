# Getting Started

This page helps you install KeepGPU, confirm the CLI can see your hardware, and
understand the minimum knobs you need to keep a GPU occupied.

## Requirements

- NVIDIA drivers + CUDA runtime visible to PyTorch.
- Python 3.9+ (matching the version in your environment/cluster image).
- Optional but recommended: `nvidia-smi` in `PATH` for utilization monitoring (CUDA) or `rocm-smi` if you install the `rocm` extra.

!!! warning "ROCm & multi-tenant clusters"
    The current release focuses on CUDA devices. ROCm/AMD support is experimental;
    controllers will raise `NotImplementedError` if CUDA is unavailable.

## Install

=== "Stable release (PyPI)"
    ```bash
    pip install keep-gpu
    ```

=== "ROCm extra (utilities for ROCm telemetry)"
    ```bash
    pip install keep-gpu[rocm]
    ```
    Install your ROCm-compatible PyTorch build separately (per upstream instructions).

=== "Editable dev install"
    ```bash
    git clone https://github.com/Wangmerlyn/KeepGPU.git
    cd KeepGPU
    pip install -e .[dev]
    ```

## Sanity check

1. Make sure PyTorch can see at least one device:
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```
   A non-zero integer indicates CUDA is available.
2. Run the CLI in dry form (press `Ctrl+C` after a few seconds):
   ```bash
   keep-gpu --interval 30 --vram 512MB
   ```
   You should see Rich logs showing the GPUs being kept awake.

If you encounter CUDA errors, run `python -m keep_gpu.benchmark` to confirm your
drivers/toolkit can allocate VRAM outside KeepGPU.

!!! tip "Lower-power keep-alive"
    KeepGPU uses intervalled elementwise ops (not big matmul floods) so you can
    keep schedulers happy while keeping power and thermals modest.

## Your first keep-alive loop

```bash
keep-gpu --interval 120 --gpu-ids 0 --vram 1GiB
```

- `--interval` controls the sleep between utilization checks (seconds).
- `--gpu-ids` limits the job to a subset of visible devices.
- `--vram` accepts human-readable sizes; KeepGPU allocates one tensor of that size.

Leave the command running while you prepare data or review notebooks. When you are
ready to hand the GPU back, hit `Ctrl+C`—controllers will release VRAM and exit.

## KeepGPU inside Python

The CLI wraps the same controllers you can import directly:

```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

with CudaGPUController(rank=0, interval=0.5, vram_to_keep="1GiB"):
    preprocess_dataset()   # Runs while GPU is pinned

train_model()              # GPU freed upon exiting the context
```

Prefer to manage multiple devices at once?

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

with GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="750MB", interval=60):
    run_cpu_bound_stage()
```

From here, jump to the CLI Playbook for scenario-driven guidance or the API
recipes if you need to embed KeepGPU in orchestration scripts.
