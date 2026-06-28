# Getting Started

This page helps you install KeepGPU, confirm the CLI can see your hardware, and
understand the minimum knobs you need to keep a GPU occupied.

## Requirements

- NVIDIA drivers + CUDA runtime visible to PyTorch.
- Python 3.9+ (matching the version in your environment/cluster image).
- Optional but recommended: `nvidia-smi` in `PATH` for utilization monitoring (CUDA) or `rocm-smi` if you install the `rocm` extra.

!!! info "Platforms"
    CUDA is the primary path; ROCm is supported by way of the `rocm` extra
    (requires a ROCm-enabled PyTorch build); Mac M series (M1/M2/M3/M4) is
    supported by way of the `macm` extra using Metal Performance Shaders (MPS).
    CPU-only environments can import the package but controllers will not start.

## Install

=== "Stable release (PyPI)"
    ```bash
    pip install keep-gpu
    ```

=== "CUDA (example: cu121)"
    ```bash
    pip install --index-url https://download.pytorch.org/whl/cu121 torch
    pip install keep-gpu
    ```

=== "ROCm (example: rocm6.1)"
    ```bash
    pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
    pip install keep-gpu[rocm]
    ```
    Install the ROCm-compatible PyTorch build that matches your runtime.

=== "CPU-only"
    ```bash
    pip install torch
    pip install keep-gpu
    ```

=== "Editable dev install"
    ```bash
    git clone https://github.com/Wangmerlyn/KeepGPU.git
    cd KeepGPU
    pip install -e .[dev]
    ```

=== "Mac M series (M1/M2/M3/M4)"
    ```bash
    pip install torch
    pip install keep-gpu[macm]
    ```
    MPS (Metal Performance Shaders) backend will be used automatically on
    Apple Silicon Macs. Note: GPU utilization monitoring is not available
    on macOS.

## Pick your interface

- **CLI** – fastest way to reserve GPUs from a shell; see [CLI Playbook](guides/cli.md).
- **Python module** – embed keep-alive loops inside orchestration code; see [Python API Recipes](guides/python.md).
- **MCP server** – expose KeepGPU over JSON-RPC (stdio or HTTP) for agents; see [MCP Server](guides/mcp.md).

## Sanity check

1. Make sure PyTorch can see at least one device:
   ```bash
   python -c "import torch; print(torch.cuda.device_count())"
   ```
   A non-zero integer indicates CUDA is available.

   On Mac M series:
   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```
   Should print `True` on Apple Silicon Macs.

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

- `--interval` controls the finite positive sleep between utilization checks
  (seconds). Values above the Python runtime wait limit are rejected.
- `--gpu-ids` limits the job to a subset of visible device ordinals. Set
  `CUDA_VISIBLE_DEVICES` before starting KeepGPU if you need physical-device
  filtering. In service mode, `keep-gpu list-gpus` and the dashboard show these
  same start-compatible visible ordinals as GPU IDs; physical metadata is only
  informational.
- `--vram` accepts human-readable sizes or bare bytes; KeepGPU allocates one
  tensor of that size. Byte-equivalent values above 1 PiB are rejected.

Leave the command running while you prepare data or review notebooks. When you are
ready to hand the GPU back, hit `Ctrl+C`—controllers will release VRAM and exit.

## Non-blocking workflow for agents

Use service mode when you need the terminal for follow-up commands:

```bash
keep-gpu start --gpu-ids 0 --interval 120 --vram 1GiB --busy-threshold 25
keep-gpu status
```

You can inspect and control sessions in a browser by starting the service and
opening:

```bash
keep-gpu serve --host 127.0.0.1 --port 8765
```

`http://127.0.0.1:8765/`

## KeepGPU inside Python

Prefer code-level control? Import the controllers directly (full recipes in
[Python API Recipes](guides/python.md)):

```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

with CudaGPUController(rank=0, interval=0.5, vram_to_keep="1GiB"):
    preprocess_dataset()   # Runs while GPU is pinned

train_model()              # GPU freed upon exiting the context
```

Direct CUDA/ROCm `rank` values are visible ordinals in the current process and
are rejected during construction if they are non-integer, negative, or outside
the visible device count.

Prefer to manage multiple devices at once?

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

with GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="750MB", interval=60):
    run_cpu_bound_stage()
```

Those `gpu_ids` are visible device ordinals after CUDA or ROCm visibility
filtering. CUDA telemetry resolves `CUDA_VISIBLE_DEVICES` and treats duplicate
or ambiguous masks as unavailable telemetry; ROCm telemetry resolves
`ROCR_VISIBLE_DEVICES` and one matching `HIP_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES` overlay before querying vendor utilization counters.

From here, jump to the CLI Playbook for scenario-driven guidance or the API
recipes if you need to embed KeepGPU in orchestration scripts.

## For contributors

Developing locally? See [Contributing](contributing.md) for dev install, test
commands (including CUDA/ROCm markers), and PR tips.
