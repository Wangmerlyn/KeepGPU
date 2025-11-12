# CLI Reference

`keep-gpu` is implemented with [Typer](https://typer.tiangolo.com/) and exposes a
single command with a handful of options. This page lists every flag plus the
environment variables you can use to customize logging.

## Synopsis

```bash
keep-gpu [OPTIONS]
```

KeepGPU blocks until you press `Ctrl+C`. When interrupted, it releases every
controller, clears the CUDA cache, and exits with status `0`.

## Options

| Option | Type | Description |
| --- | --- | --- |
| `--interval INTEGER` | seconds | Sleep duration between utilization checks and keep-alive batches. Lower values keep the GPU hotter; higher values save power. Default: `300`. |
| `--gpu-ids TEXT` | comma-separated ints | Subset of GPUs to guard (for example, `0,2`). If omitted, KeepGPU enumerates `torch.cuda.device_count()` and protects every visible device. |
| `--vram TEXT` | human size or bytes | Amount of memory each GPU controller allocates. Accept formats like `512MB`, `1GiB`, or `1073741824`. Default: `1GiB`. |
| `--busy-threshold INTEGER` / `--util-threshold INTEGER` | percent | Upper bound on observed utilization. When NVML reports a higher number, the controller adds extra sleeps so it will not interfere with legitimate workloads. Default: `-1` (never throttle). |
| `--help` | flag | Show Typer-generated help and exit. |

!!! tip "Choosing a VRAM value"
    Allocating 20‑30 % of a GPU’s memory is usually enough for schedulers that
    only watch the “memory in use” column. If your cluster monitors utilization,
    pair a higher `--vram` with a shorter `--interval`.

!!! note "`--threshold` legacy flag"
    Older scripts may still pass `--threshold`. Numeric values map to
    `--busy-threshold`; strings such as `1GiB` override `--vram`. Prefer the
    explicit flags going forward.

## Environment variables

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA filtering. The CLI respects it and then applies `--gpu-ids` on top. |
| `CONSOLE_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `no`/`0` to silence console logs entirely. |
| `FILE_LOG_LEVEL` | Same accepted values; when set, KeepGPU writes timestamped log files under `./logs/`. |

Set them inline:

```bash
CONSOLE_LOG_LEVEL=DEBUG FILE_LOG_LEVEL=INFO keep-gpu --interval 30
```

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Normal completion or user-triggered `Ctrl+C`. |
| `1` | Input validation error (for example, malformed `--gpu-ids`). |
| `>1` | Unhandled exception. Check the Rich traceback or log files. |

Need more detail on the Python API and controllers backing the CLI? See the
[API reference](api.md) next.
