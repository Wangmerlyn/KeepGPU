---
name: gpu-keepalive-with-keepgpu
description: Install and operate KeepGPU for GPU keep-alive with both blocking CLI and non-blocking service workflows. Use when users ask for keep-gpu command construction, start/status/stop session control, dashboard usage, tuning (--vram, --interval, --busy-threshold), installation from this repository, or troubleshooting keep sessions; do not use for repository development, code refactoring, or unrelated Python tooling.
---

# KeepGPU CLI Operator

Use this workflow to run `keep-gpu` safely and effectively.

## Prerequisites

- Confirm at least one GPU is visible (`python -c "import torch; print(torch.cuda.device_count())"`).
- Run commands in a shell where CUDA/ROCm drivers are already available.
- Use `Ctrl+C` to stop KeepGPU and release memory cleanly.

## Install KeepGPU

Install PyTorch first for your platform, then install KeepGPU.

### Option A: Install from package index

```bash
# CUDA example (change cu121 to your CUDA version)
pip install --index-url https://download.pytorch.org/whl/cu121 torch
pip install keep-gpu
```

```bash
# ROCm example (change rocm6.1 to your ROCm version)
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
pip install keep-gpu[rocm]
```

### Option B: Install directly from Git URL (no local clone)

Prefer this option when users only need the CLI and do not need local source edits. This avoids checkout directory and cleanup overhead.

```bash
pip install "git+https://github.com/Wangmerlyn/KeepGPU.git"
```

If SSH access is configured:

```bash
pip install "git+ssh://git@github.com/Wangmerlyn/KeepGPU.git"
```

ROCm variant from Git URL:

```bash
pip install "keep_gpu[rocm] @ git+https://github.com/Wangmerlyn/KeepGPU.git"
```

### Option C: Install from a local source checkout (explicit path)

Use this option only when users already have a local checkout or plan to edit source.

```bash
git clone https://github.com/Wangmerlyn/KeepGPU.git
cd KeepGPU
pip install -e .
```

If the checkout already exists somewhere else, install by absolute path:

```bash
pip install -e /absolute/path/to/KeepGPU
```

For ROCm users from local checkout:

```bash
pip install -e ".[rocm]"
```

Verify installation:

```bash
keep-gpu --help
```

## Command model

KeepGPU supports two execution modes.

### Blocking mode (compatibility)

```bash
keep-gpu --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25
```

Use when users intentionally want one foreground process and manual `Ctrl+C` stop.

### Non-blocking mode (recommended for agents)

```bash
keep-gpu start --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25
keep-gpu status
keep-gpu stop --all
keep-gpu service-stop
```

`start` auto-starts local service when unavailable.

`Ctrl+C` stops only foreground blocking runs. For service mode sessions started by `keep-gpu start`, use `keep-gpu status`, `keep-gpu stop`, and `keep-gpu service-stop`.

CLI options to tune:

- `--gpu-ids`: comma-separated IDs (`0`, `0,1`). If omitted, KeepGPU uses all visible GPUs.
- `--vram`: VRAM to hold per GPU (`512MB`, `1GiB`, or raw bytes).
- `--interval`: seconds between keep-alive cycles.
- `--busy-threshold` (`--util-threshold` alias): if utilization is above this percent, KeepGPU backs off.

Legacy compatibility:

- `--threshold` is deprecated but still accepted.
- Numeric `--threshold` maps to busy threshold.
- String `--threshold` maps to VRAM.

## Agent workflow

1. Collect workload intent: target GPUs, hold duration, and whether node is shared.
2. Choose mode:
   - blocking mode for manual shell sessions,
   - non-blocking mode for agent pipelines (default recommendation).
3. Choose safe defaults when unspecified: `--vram 1GiB`, `--interval 60-120`, `--busy-threshold 25`.
4. Provide command sequence with verification and stop command.
5. For non-blocking mode, include `status`, `stop`, and daemon shutdown (`service-stop`).

## Command templates

Single GPU while preprocessing (blocking):

```bash
keep-gpu --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25
```

All visible GPUs with lighter load (blocking):

```bash
keep-gpu --vram 512MB --interval 180
```

Agent-friendly non-blocking sequence:

```bash
keep-gpu start --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25
keep-gpu status
keep-gpu stop --job-id <job_id>
keep-gpu service-stop
```

Open dashboard:

```text
http://127.0.0.1:8765/
```

Remote sessions (preferred: `tmux` for visibility and control):

```bash
tmux new -s keepgpu
keep-gpu --gpu-ids 0 --vram 1GiB --interval 300
# Detach with Ctrl+b then d; reattach with: tmux attach -t keepgpu
```

Fallback when `tmux` is unavailable:

```bash
nohup keep-gpu --gpu-ids 0 --vram 1GiB --interval 300 > keepgpu.log 2>&1 &
echo $! > keepgpu.pid
# Monitor: tail -f keepgpu.log
# Stop: kill "$(cat keepgpu.pid)"
```

## Troubleshooting

- Invalid `--gpu-ids`: ensure comma-separated integers only.
- Allocation failure / OOM: reduce `--vram` or free memory first.
- No utilization telemetry: ensure `nvidia-ml-py` works and `nvidia-smi` is available.
- No GPUs detected: verify drivers, CUDA/ROCm runtime, and `torch.cuda.device_count()`.

## Example

User request: "Install KeepGPU from GitHub and keep GPU 0 alive while I preprocess."

Suggested response shape:

1. Install: `pip install "git+https://github.com/Wangmerlyn/KeepGPU.git"`
2. Run: `keep-gpu start --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25`
3. Verify: `keep-gpu status` or dashboard `http://127.0.0.1:8765/`; stop session with `keep-gpu stop --job-id <job_id>` and daemon with `keep-gpu service-stop`.

## Limitations

- KeepGPU is not a scheduler; it only keeps already accessible GPUs active.
- KeepGPU behavior depends on cluster policy; some schedulers require higher VRAM or tighter intervals.
