# CLI Playbook

Practical examples for running `keep-gpu` on shared clusters, workstations, and
Jupyter environments.

## Command anatomy

```bash
keep-gpu --interval 120 --gpu-ids 0,1 --vram 2GiB --threshold 25
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--interval` | Sleep between keep-alive cycles (seconds). Lower = tighter lock. | `300` |
| `--gpu-ids` | Comma-separated visible IDs. Leave unset to keep every detected GPU busy. | all |
| `--vram` | Amount of memory each controller allocates. Accepts `800MB`, `1GiB`, `1073741824`, etc. | `1GiB` |
| `--threshold` | Skip work when utilization is already above this percentage. | `-1` (never skip) |

!!! info "What happens under the hood?"
    Each GPU gets a `CudaGPUController` that allocates one tensor sized by
    `--vram` and runs a lightweight matmul loop. Controllers watch `nvidia-smi`
    to avoid hogging a device that is already busy (see `--threshold`).

## Scenarios

### 1. Hold a single GPU while preprocessing

```bash
keep-gpu --gpu-ids 0 --interval 60 --vram 2GiB
```

- Keeps card `0` alive with moderate VRAM allocation.
- Suitable for local experiments where you just need the scheduler to see
  sustained activity.

### 2. Park every GPU on the node overnight

```bash
keep-gpu --interval 180 --vram 512MB
```

- Launch inside `tmux`/`screen` or your cluster’s “interactive session.”
- Use a long interval to reduce background load while still holding the cards.

### 3. Share the node without starving teammates

```bash
keep-gpu --gpu-ids 0,1 --interval 90 --threshold 35
```

- Controllers pause their work whenever utilization exceeds 35%.
- Lets you reserve GPUs 0 and 1 while GPUs 2+ remain untouched.

### 4. Run from Jupyter or VS Code terminal

```bash
!keep-gpu --interval 45 --vram 768MB --threshold 50
```

- Prefix with `!` (Jupyter) or use the integrated terminal.
- Stop with `Ctrl+C` when you are ready to start the actual CUDA workload.

### 5. Background job via scheduler

```bash
nohup keep-gpu --interval 300 --gpu-ids 0 --vram 1GiB > keepgpu.log 2>&1 &
```

- `nohup` keeps the process alive after you disconnect.
- Combine with your cluster’s reservation commands (e.g., `srun`, `bsub`, `qsub`).

## Observability and safety

- **Logging levels** – Set `CONSOLE_LOG_LEVEL=DEBUG` or `FILE_LOG_LEVEL=INFO`
  to capture detailed metrics. Logs land in `logs/*.log`.
- **VRAM tuning** – Start with 1–2 GiB. Some schedulers only inspect “memory in
  use,” so going below 500 MB may not fool them.
- **Graceful exit** – Use `Ctrl+C`; KeepGPU releases tensors and clears the CUDA
  cache so the next job starts with a clean slate.
- **Failure recovery** – If allocation fails (e.g., VRAM already full), the CLI
  retries after `--interval` and logs the error. Adjust `--vram` downwards or
  free memory manually.

Ready to embed this behavior inside your training scripts? Head to the
[Python API Recipes](python.md) next.
