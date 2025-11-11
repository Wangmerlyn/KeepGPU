# KeepGPU

> Keep a GPU busy so the scheduler, lab mate, or cluster watchdog does not take it away.

KeepGPU is a tiny-but-focused toolkit that continuously allocates VRAM and launches
low-cost CUDA workloads so that a GPU always looks “in use.” You can run it as a CLI
while you prep data, or embed the controllers directly in Python to guard resources
during longer CPU-bound sections of your workflow.

## Why it matters

- **Hold your reservation** – Prevent preemptive job eviction on shared clusters that
  recycle idle GPUs after a short grace period.
- **Avoid surprise card sharing** – Keep teammates, notebooks, or background jobs from
  silently grabbing the GPU while you are still working.
- **Stay lightweight** – Instead of pinning a full training job, KeepGPU runs a tiny
  CUDA matmul loop and sleeps between bursts to keep thermals/noise low.

## What’s inside

- Rich CLI based on Typer + Rich (interval tuning, selective GPU IDs, VRAM budgeting).
- `GlobalGPUController` that spins up a keep-alive worker per GPU.
- `CudaGPUController` context manager for fine-grained orchestration inside scripts.
- Helpers for parsing human VRAM sizes (`1GiB`, `850MB`, etc.) and platform detection.

## Where to go next

- :material-rocket-launch: **[Getting Started](getting-started.md)** – Install,
  verify `keep-gpu` works, and run your first protection loop.
- :material-console-line: **[CLI Playbook](guides/cli.md)** – Task-focused recipes
  for pinning cards on clusters, workstations, or Jupyter.
- :material-code-tags: **[Python API Recipes](guides/python.md)** – Drop-in snippets
  for wrapping preprocessing stages or orchestration scripts.
- :material-diagram-project: **[How KeepGPU Works](concepts/architecture.md)** –
  Learn how controllers allocate VRAM and throttle themselves.
- :material-book-open-outline: **[Reference](reference/cli.md)** – Full option list
  plus mkdocstrings API reference.

!!! tip "Prefer a fast skim?"
    The left sidebar mirrors the lifecycle: overview → guides → concepts →
    references. Jump straight to what you need; sections stand on their own.
