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
| `--interval INTEGER` | seconds | Finite positive sleep duration between utilization checks and keep-alive batches; values above the Python runtime wait limit are rejected. |
| `--gpu-ids TEXT` | comma-separated unique non-negative ints | Subset of visible device ordinals to guard (for example, `0,2`). Omit to use all visible GPUs; startup fails if that resolves to none or if an explicit ordinal is out of range. |
| `--vram TEXT` | human size or bare bytes | Amount of memory each GPU controller allocates (`512MB`, `1GiB`, `1073741824`); byte-equivalent values above 1 PiB are rejected. |
| `--busy-threshold INTEGER` / `--util-threshold INTEGER` | percent | `0..100` backs off before allocation/compute when utilization is above this value or unavailable; `-1` disables utilization backoff. |
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

Local input validation runs before service auto-start. Invalid `--vram`,
`--job-id`, `--interval`, `--busy-threshold`, or `--gpu-ids` values fail before
daemon startup or RPC.

| Option | Default | Description |
| --- | --- | --- |
| `--gpu-ids` | all | Comma-separated unique visible device ordinals in the service process environment. |
| `--vram` | `1GiB` | Per-GPU keep memory target; byte-equivalent values above 1 PiB are rejected. |
| `--interval` | `300` | Finite positive keep cycle interval in seconds, capped by the Python runtime wait limit. |
| `--busy-threshold` / `--util-threshold` | `25` | `0..100` backs off when utilization is above this value or telemetry is unavailable; `-1` disables utilization backoff. |
| `--job-id` | auto | Optional URL-path-safe custom id. Invalid IDs are rejected locally before service auto-start; valid IDs must be unique across active and starting sessions. |
| `--host` | `127.0.0.1` | Service host to contact. |
| `--port` | `8765` | Service port to contact. |
| `--auto-start/--no-auto-start` | `--auto-start` | Auto-start local service if unavailable. |

### `keep-gpu status`

| Option | Description |
| --- | --- |
| `--job-id` | Optional session id; omit to list all tracked sessions, including in-progress starts, in-progress releases, or failed releases. |
| `--host`, `--port` | Service host/port. |

Prints a directly parseable JSON object, including `{"error": "..."}` for
service/runtime errors after CLI parsing succeeds. Malformed JSON-RPC service
envelopes are reported as JSON error objects instead of empty success results.
Started sessions with terminal worker allocation/runtime failures remain listed
as `state="runtime_failed"` with `last_error` and can still be stopped. Normal
busy-GPU or unavailable-telemetry backoff keeps the session active; it is not a
runtime failure.

### `keep-gpu stop`

| Option | Description |
| --- | --- |
| `--job-id` | Stop one session. |
| `--all` | Stop all sessions. |
| `--host`, `--port` | Service host/port. |

`--job-id` and `--all` are mutually exclusive. Passing both returns a JSON
error before any RPC or stop-all fallback runs.
Stop waits for in-progress starts to settle before returning `not found` or
taking the stop-all snapshot, so starting sessions are not silently skipped.
For `--all`, starts that begin after that command's initial snapshot are not
stopped by that command.
`--all` releases the sessions in its snapshot concurrently and prints results
in deterministic snapshot order with the same additive response fields.
The output is a directly parseable JSON object, including `{"error": "..."}` for
service/runtime errors after CLI parsing succeeds. Malformed JSON-RPC service
envelopes are reported as JSON error objects instead of empty success results.

### `keep-gpu list-gpus`

Returns GPU telemetry from service. `id` and `visible_id` are the visible
ordinals accepted by `--gpu-ids` and service `gpu_ids`; optional `physical_id`
or `uuid` fields are metadata only. The output is a directly parseable JSON
object, including `{"error": "..."}` for service/runtime errors after CLI
parsing succeeds. Malformed JSON-RPC service envelopes are reported as JSON
error objects instead of empty success results.

### `keep-gpu service-stop`

Stops the ownership-verified local daemon process created by auto-start logic.

| Option | Description |
| --- | --- |
| `--host`, `--port` | Service host/port. |
| `--force` | Skip active-session RPC checks and stop the daemon only if the auto-start ownership record verifies the process. |

## Service HTTP endpoints

Only omitted/`null` `job_id` values mean generated IDs or all-sessions, depending
on the method. Custom IDs must be non-empty strings containing only letters,
digits, `.`, `_`, `-`, or `~`; invalid REST path IDs return `400` before acting.
Supported REST route/method errors are JSON objects: validation errors return
`400`, unknown API routes return `404`, and unexpected service/runtime failures
return `500` instead of dropping the connection.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service liveness probe. |
| `/api/gpus` | GET | GPU telemetry (`id`/`visible_id` are start-compatible visible ordinals; optional `physical_id`/`uuid` are metadata; unsupported fields are `null`). |
| `/api/sessions` | GET | Tracked keep sessions, including `state="starting"` during startup, `state="runtime_failed"` plus `last_error` for retained worker failures, and `state`/`last_error` for in-progress or failed stops. |
| `/api/sessions/{job_id}` | GET | One session status, including `state` and `last_error` when active, starting, runtime-failed, or retained after stop problems. |
| `/api/sessions` | POST | Start session with a JSON object body (`gpu_ids`, `vram`, finite positive bounded `interval`, `busy_threshold`, `job_id`); `vram` accepts human sizes or bytes up to 1 PiB byte-equivalent, omitted `gpu_ids` means all GPUs visible to the service process, omitted `busy_threshold` uses `25`, and empty, duplicate, or out-of-range selections are invalid. |
| `/api/sessions` | DELETE | Stop all sessions; returns `stopped`, `timed_out`, `failed`, and `errors`. |
| `/api/sessions/{job_id}` | DELETE | Stop one session; returns `stopped`, `timed_out`, `failed`, and `errors`. |
| `/rpc` | POST | JSON-RPC compatibility endpoint. |
| `/` | GET | Dashboard UI. |

## Environment variables

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA filtering, and a HIP-compatible ROCm overlay when `HIP_VISIBLE_DEVICES` is unset. Set it before starting KeepGPU; `--gpu-ids` selects visible ordinals after filtering, and service mode uses the daemon process environment. CUDA utilization telemetry maps visible ordinals through numeric or UUID tokens before querying NVML, and duplicate or unresolved mappings report unavailable telemetry. ROCm utilization treats this as one overlay on top of `ROCR_VISIBLE_DEVICES` and reports unavailable telemetry when it conflicts with `HIP_VISIBLE_DEVICES`. |
| `ROCR_VISIBLE_DEVICES` | ROCm base visibility mask. KeepGPU keeps `gpu_ids` as visible ordinals while resolving this mask before querying ROCm SMI telemetry. Unsupported, malformed, duplicate, or out-of-range numeric masks report unavailable utilization instead of guessing a physical SMI index. |
| `HIP_VISIBLE_DEVICES` | ROCm HIP-layer visibility overlay. If both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set, they must describe the same numeric overlay for ROCm telemetry to query ROCm SMI; otherwise utilization is unavailable and non-negative `busy_threshold` values sleep for that cycle. |
| `CONSOLE_LOG_LEVEL` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `no`). |
| `FILE_LOG_LEVEL` | File log level; writes logs under `./logs/` when enabled. |
