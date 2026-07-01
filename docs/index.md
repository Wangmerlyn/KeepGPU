# KeepGPU

> Hold a GPU reservation politely while the scheduler, lab mate, or cluster watchdog waits.

KeepGPU is a tiny-but-focused toolkit that allocates memory and runs low-cost
platform-specific keep-alive work when utilization backoff permits. When
utilization telemetry is unavailable, the default loop backs off and sleeps
instead of forcing compute; `--busy-threshold -1` is the explicit unconditional
keepalive mode, especially on Mac M/MPS.

## Why it matters

- **Hold your reservation** – Prevent preemptive job eviction on shared clusters that
  recycle idle GPUs after a short grace period.
- **Avoid surprise card sharing** – Keep teammates, notebooks, or background jobs from
  silently grabbing the GPU while you are still working.
- **Stay lightweight** – Instead of pinning a full training job, KeepGPU runs
  periodic lightweight elementwise ops and sleeps between bursts to keep
  thermals/noise low.

## What’s inside

- Rich CLI based on Typer + Rich for blocking and non-blocking session
  workflows.
- `GlobalGPUController` that spins up a keep-alive worker per selected GPU.
- `CudaGPUController`, `RocmGPUController`, and `MacMGPUController` context
  managers for fine-grained orchestration inside scripts.
- Service dashboard, REST endpoints, JSON-RPC methods, and MCP server for
  browser controls, scripts, and agent integrations.
- Helpers for parsing human VRAM sizes (`1GiB`, `850MB`, etc.) and platform detection.
- Power-aware keep-alive loop: periodic elementwise ops to signal “busy” without flooding matmuls or spiking thermals.

## Where to go next

- :material-rocket-launch: **[Getting Started](getting-started.md)** – Install,
  verify `keep-gpu` works, and run your first protection loop.
- :material-console-line: **[CLI Playbook](guides/cli.md)** – Task-focused recipes
  for pinning cards on clusters, workstations, or Jupyter.
- :material-code-tags: **[Python API Recipes](guides/python.md)** – Drop-in snippets
  for wrapping preprocessing stages or orchestration scripts.
- :material-monitor-dashboard: **[Dashboard + API](guides/mcp.md)** – Run
  `keep-gpu serve` and open `http://127.0.0.1:8765/` for session controls and
  telemetry.
- :material-lan: **[MCP and Service API](guides/mcp.md)** – JSON-RPC + REST endpoints
  for agents and remote orchestration.
- :material-diagram-project: **[How KeepGPU Works](concepts/architecture.md)** –
  Learn how controllers allocate VRAM and throttle themselves.
- :material-book-open-outline: **[Reference](reference/cli.md)** – Full option list
  plus mkdocstrings API reference.

!!! tip "Prefer a fast skim?"
    The left sidebar mirrors the lifecycle: overview → usage → concepts →
    references. Jump straight to what you need; sections stand on their own.
