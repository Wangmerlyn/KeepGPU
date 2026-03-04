# CLI Reference

`keep-gpu` supports both blocking mode and service-driven non-blocking mode.

## Command summary

```bash
keep-gpu [blocking options]
keep-gpu serve [--host 127.0.0.1] [--port 8765]
keep-gpu start [options]
keep-gpu status [--job-id ID] [--host 127.0.0.1] [--port 8765]
keep-gpu stop (--job-id ID | --all)
keep-gpu service-stop [--host 127.0.0.1] [--port 8765] [--force]
keep-gpu list-gpus
```

## Blocking mode options

These options apply when you run `keep-gpu` without subcommands.

| Option | Type | Description |
| --- | --- | --- |
| `--interval INTEGER` | seconds | Sleep duration between utilization checks and keep-alive batches. |
| `--gpu-ids TEXT` | comma-separated ints | Subset of GPUs to guard (for example, `0,2`). Omit to use all visible GPUs. |
| `--vram TEXT` | human size or bytes | Amount of memory each GPU controller allocates (`512MB`, `1GiB`, `1073741824`). |
| `--busy-threshold INTEGER` / `--util-threshold INTEGER` | percent | Back off when utilization is above this value. |
| `--threshold TEXT` | deprecated | Legacy alias: numeric values map to busy-threshold, size strings map to vram. |

## Service mode

### `keep-gpu serve`

Starts local KeepGPU service (HTTP + JSON-RPC + dashboard).

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Service bind host. |
| `--port` | `8765` | Service port. |

### `keep-gpu start`

Starts a keep session and returns immediately with `job_id`.

| Option | Default | Description |
| --- | --- | --- |
| `--gpu-ids` | all | Comma-separated GPU IDs. |
| `--vram` | `1GiB` | Per-GPU keep memory target. |
| `--interval` | `300` | Keep cycle interval in seconds. |
| `--busy-threshold` / `--util-threshold` | `-1` | Backoff threshold. |
| `--job-id` | auto | Optional custom id. |
| `--host` | `127.0.0.1` | Service host to contact. |
| `--port` | `8765` | Service port to contact. |
| `--auto-start/--no-auto-start` | `--auto-start` | Auto-start local service if unavailable. |

### `keep-gpu status`

| Option | Description |
| --- | --- |
| `--job-id` | Optional session id; omit to list all active jobs. |
| `--host`, `--port` | Service host/port. |

### `keep-gpu stop`

| Option | Description |
| --- | --- |
| `--job-id` | Stop one session. |
| `--all` | Stop all sessions. |
| `--host`, `--port` | Service host/port. |

### `keep-gpu list-gpus`

Returns GPU telemetry from service.

### `keep-gpu service-stop`

Stops local daemon process created by auto-start logic.

| Option | Description |
| --- | --- |
| `--host`, `--port` | Service host/port. |
| `--force` | Stop daemon even if active sessions exist. |

## Service HTTP endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service liveness probe. |
| `/api/gpus` | GET | GPU telemetry (`id`, `name`, memory, utilization). |
| `/api/sessions` | GET | Active keep sessions. |
| `/api/sessions/{job_id}` | GET | One session status. |
| `/api/sessions` | POST | Start session (`gpu_ids`, `vram`, `interval`, `busy_threshold`, `job_id`). |
| `/api/sessions` | DELETE | Stop all sessions. |
| `/api/sessions/{job_id}` | DELETE | Stop one session. |
| `/rpc` | POST | JSON-RPC compatibility endpoint. |
| `/` | GET | Dashboard UI. |

## Environment variables

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA filtering. Blocking mode honors it before `--gpu-ids`. |
| `CONSOLE_LOG_LEVEL` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `no`). |
| `FILE_LOG_LEVEL` | File log level; writes logs under `./logs/` when enabled. |
