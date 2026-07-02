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
| `--interval NUMBER` | seconds | Finite positive sleep duration, including fractional values, between utilization checks and keep-alive batches; values above the Python runtime wait limit are rejected. |
| `--gpu-ids TEXT` | comma-separated unique non-negative ints | Subset of visible device ordinals to guard (for example, `0,2`). Omit to let the controller resolve all visible GPUs; explicit empty or whitespace-only values are invalid. Startup fails if all-visible resolution finds no GPUs or if an explicit ordinal is out of range. |
| `--vram TEXT` | human size or bare bytes | Amount of memory each GPU controller allocates (`512MB`, `1GiB`, `1073741824`); byte-equivalent values above 1 PiB are rejected. |
| `--busy-threshold INTEGER` / `--util-threshold INTEGER` | percent | `0..100` backs off before allocation/compute when utilization is above this value or unavailable; `-1` disables utilization backoff. |
| `--threshold TEXT` | deprecated | Legacy alias: numeric values map to busy-threshold, size strings map to vram. |

## Service mode

### `keep-gpu serve`

Starts local KeepGPU service (HTTP + JSON-RPC + dashboard).

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Service bind host. Must be a DNS hostname or IPv4 address. |
| `--port` | `8765` | Service port as a plain ASCII decimal integer in `1..65535`. |

### `keep-gpu start`

Starts a keep session and returns immediately with `job_id`.

Local input validation runs before service auto-start. Invalid `--vram`,
`--job-id`, `--interval`, `--busy-threshold`, `--gpu-ids`, `--host`, or `--port`
values fail before daemon startup or RPC. Omit `--gpu-ids` to use all visible
GPUs; explicit empty or whitespace-only values are invalid.
If `start` auto-starts the service and the service then reports expected
startup unavailability before creating a session, the CLI best-effort stops the
just-created daemon instead of leaving it idle.
Malformed JSON-RPC service envelopes, including missing/non-integer
`error.code` or missing/non-string `error.message`, are response errors and do
not trigger this rollback.
When `--job-id` is supplied, the successful `start_keep` response must echo the
requested `job_id` or the response is rejected as malformed.
The same best-effort cleanup runs when auto-start times out before the service
passes its health check. Auto-start refuses to overwrite an ownership-verified
live daemon PID record when that daemon's health endpoint is unavailable; inspect
the service log at
`~/.keepgpu/service-<host-with-dots-as-underscores>-<port>.log` or run
`keep-gpu service-stop --force` before retrying.

| Option | Default | Description |
| --- | --- | --- |
| `--gpu-ids` | all | Comma-separated unique visible device ordinals in the service process environment. Omit for all visible GPUs; empty or whitespace-only values are invalid. |
| `--vram` | `1GiB` | Per-GPU keep memory target; byte-equivalent values above 1 PiB are rejected. |
| `--interval` | `300` | Finite positive keep cycle interval in seconds, including fractional values, capped by the Python runtime wait limit. |
| `--busy-threshold` / `--util-threshold` | `25` | `0..100` backs off when utilization is above this value or telemetry is unavailable; `-1` disables utilization backoff. |
| `--job-id` | auto | Optional URL-path-safe custom id. Invalid IDs are rejected locally before service auto-start; valid IDs must be unique across active and starting sessions. |
| `--host` | `127.0.0.1` | Service host to contact; invalid values are rejected before auto-start. |
| `--port` | `8765` | Service port to contact; must be a plain ASCII decimal integer in `1..65535`. |
| `--auto-start/--no-auto-start` | `--auto-start` | Auto-start local service if unavailable. |

### `keep-gpu status`

| Option | Description |
| --- | --- |
| `--job-id` | Optional non-empty URL-path-safe session id; omit to list all tracked sessions, including in-progress starts, in-progress releases, or failed releases. Invalid explicit IDs are rejected locally before RPC. |
| `--host`, `--port` | Service host/port. Invalid endpoint values, including non-integer or out-of-range ports, are rejected locally and printed as JSON errors before RPC. |

Prints a directly parseable JSON object, including `{"error": "..."}` for
service/runtime errors after CLI parsing succeeds. Malformed JSON-RPC service
envelopes, including missing/non-integer `error.code` or missing/non-string
`error.message`, and malformed status job records are reported as JSON error
objects instead of empty success results.
When `--job-id` is supplied, the returned `job_id` must match the requested
target or the response is rejected as malformed.
Status record `state` values are validated against the known lifecycle states:
`active`, `starting`, `stopping`, `runtime_failed`, and `stop_failed`.
The machine JSON stream is plain JSON without Rich color or highlighting, even
when the command runs under a pseudo-TTY or forced-color terminal.
Started sessions with terminal worker allocation/runtime failures remain listed
as `state="runtime_failed"` with `last_error` and can still be stopped. Normal
busy-GPU or unavailable-telemetry backoff keeps the session active; it is not a
runtime failure.

### `keep-gpu stop`

| Option | Description |
| --- | --- |
| `--job-id` | Stop one non-empty URL-path-safe session id. Invalid explicit IDs are rejected locally before RPC or stop-all fallback. |
| `--all` | Stop all sessions. |
| `--host`, `--port` | Service host/port. Invalid endpoint values, including non-integer or out-of-range ports, are rejected locally and printed as JSON errors before RPC or stop-all fallback. |

`--job-id` and `--all` are mutually exclusive. Passing both returns a JSON
error before any RPC or stop-all fallback runs.
Targeted stop waits for a matching in-progress start to settle before returning
`not found`, so starting sessions are not silently skipped. That wait is
bounded; if startup does not settle in time, the response includes the job in
`timed_out`, status shows the remembered cancellation as `state="stopping"`
with the timeout message, and the service releases a later successful startup
in the background. For `--all`, the service records the initial active/starting
boundary first, waits only for starting jobs in that boundary, and does not stop
later starts.
`--all` releases the sessions in its snapshot concurrently and prints results
in deterministic snapshot order with the same additive response fields.
If the stop RPC transport is unreachable, `--all` may force-stop an
ownership-verified local daemon. Application/runtime errors from the service are
reported as JSON errors and do not trigger daemon stop fallback based on message
text.
The output is a directly parseable JSON object, including `{"error": "..."}` for
service/runtime errors after CLI parsing succeeds. Malformed JSON-RPC service
envelopes, including missing/non-integer `error.code` or missing/non-string
`error.message`, and malformed stop result records are reported as JSON error
objects instead of empty success results. When `--job-id` is supplied, all
returned outcome and error job IDs must match the requested target.
The machine JSON stream is plain JSON without Rich color or highlighting, even
when the command runs under a pseudo-TTY or forced-color terminal.

### `keep-gpu list-gpus`

Returns GPU telemetry from service. `id` and `visible_id` are matching
non-negative, unique visible ordinals accepted by `--gpu-ids` and service
`gpu_ids`; optional `physical_id` or `uuid` fields are metadata only. The output
is a directly parseable JSON object. On CUDA, NVML records are listed only when
Torch CUDA can start the same visible ordinal set; NVML-only devices are hidden
rather than advertised as
usable `gpu_ids`. On ROCm, listed records are limited to visible ordinals that
Torch can select; nullable memory fields mean memory telemetry is unavailable
after selection succeeds. Service/runtime errors after CLI parsing succeeds are
reported as `{"error": "..."}`. CUDA/ROCm visible-device enumeration failures
are reported as startup-unavailable errors instead of successful empty GPU
lists. Malformed JSON-RPC service envelopes, including missing/non-integer
`error.code` or missing/non-string `error.message`, are reported as JSON error
objects instead of empty success results; malformed GPU records are reported the
same way. `utilization` is
either `null` or a finite number from `0` to `100`;
out-of-range telemetry is unavailable, not idle. Memory fields are
non-negative integers or `null`; invalid counters and impossible
`memory_used > memory_total` pairs are unavailable telemetry, not displayed
usage.
The machine JSON stream is plain JSON without Rich color or highlighting, even
when the command runs under a pseudo-TTY or forced-color terminal.
Invalid endpoint values, including non-integer or out-of-range ports, are
reported as JSON errors before RPC.

### `keep-gpu service-stop`

Stops the ownership-verified local daemon process created by auto-start logic.
Invalid endpoint values are rejected locally before service checks or
ownership-verified stop operations. Non-force shutdown requires the service to
be reachable, `stop_keep` to report no stopped, timed-out, or failed sessions
and no non-empty message, and a final status check to show no active sessions
before the daemon process is signaled. Malformed PID records with float or
boolean numeric identity values are ignored rather than coerced before
signaling. Auto-start cleans up and fails if it cannot create a trustworthy
ownership record for the daemon it just spawned. On systems without `/proc`,
KeepGPU may recover daemon identity from platform process metadata, but it still
signals only when recovered identity is known and exactly matches the stored
ownership record.

| Option | Description |
| --- | --- |
| `--host`, `--port` | Service host/port. `--host` must be a DNS hostname or IPv4 address, and `--port` must be a plain ASCII decimal integer in `1..65535`. |
| `--force` | Skip session RPC checks and stop the daemon only if the auto-start ownership record verifies the process. |

## Service HTTP endpoints

Only omitted/`null` `job_id` values mean generated IDs or all-sessions, depending
on the method. Custom IDs must be non-empty strings containing only letters,
digits, `.`, `_`, `-`, or `~`, and may not be standalone `.` or `..`; invalid
REST path IDs return `400` before acting.
Supported REST route/method errors are JSON objects: validation errors return
`400`, unknown API routes return `404`, expected startup-unavailable session
creation returns `503`, and unexpected service/runtime failures return `500`
instead of dropping the connection. Encoded noncanonical spellings such as
`/api%2Fsessions`, `/api%3Bdebug`, or `/api%3Fsessions` are treated as
unknown API routes and return JSON `404` responses rather than the dashboard
HTML shell. Raw leading-double-slash API aliases such as `//api/sessions` are
also unknown routes and do not start or stop sessions.
Exact API collection routes such as `/api/gpus` and `/api/sessions` do not
accept query strings or path parameters unless documented; query-shaped
collection URLs return JSON `404` responses.
The dashboard consumes those JSON objects and displays `error.message` when it is
present, falling back to the plain response body or HTTP status only when needed.
Direct JSON-RPC service calls report the same expected hardware/platform
startup-unavailable conditions, including failed CUDA/ROCm visible-device
enumeration and unavailable PyTorch MPS backends, with error code `-32000`;
arbitrary runtime failures remain `-32603 Internal error`. CLI service commands
send explicit JSON-RPC 2.0
request envelopes to `/rpc`; omitted-version direct calls are retained only for
legacy local scripts.
Omitted direct-call `params` are treated as `{}` for compatibility, but explicit
`params: null` or any other non-object `params` value returns `-32602 Invalid
params` before method side effects.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Service liveness probe. |
| `/api/gpus` | GET | GPU telemetry (`id`/`visible_id` are start-compatible visible ordinals; optional `physical_id`/`uuid` are metadata; unselectable CUDA/ROCm records are omitted; unsupported fields are `null`; expected enumeration unavailability is a structured `503`; the dashboard treats unavailable utilization as `n/a`, not `0%`). |
| `/api/sessions` | GET | Tracked keep sessions, including `state="starting"` during startup, `state="runtime_failed"` plus `last_error` for retained worker failures, and `state`/`last_error` for in-progress or failed stops. |
| `/api/sessions/{job_id}` | GET | One session status, including `state` and `last_error` when active, starting, runtime-failed, or retained after stop problems. |
| `/api/sessions` | POST | Start session with a JSON object body (`gpu_ids`, `vram`, finite positive bounded `interval`, `busy_threshold`, `job_id`); `vram` accepts human sizes or bytes up to 1 PiB byte-equivalent, omitted `gpu_ids` means all GPUs visible to the service process, omitted `busy_threshold` uses `25`, and duplicate, out-of-range, or selections with more than 64 entries are invalid. Explicit `gpu_ids` are checked against a validated `list_gpus()` response; an explicit empty `gpu_ids` list is invalid, an empty validated GPU listing is `503`, and malformed listing payloads are structured `500` errors before any session starts. |
| `/api/sessions` | DELETE | Stop all sessions; returns `stopped`, `timed_out`, `failed`, and `errors`. |
| `/api/sessions/{job_id}` | DELETE | Stop one session; returns `stopped`, `timed_out`, `failed`, and `errors`. |
| `/rpc` | POST | Exact JSON-RPC compatibility endpoint; noncanonical `/rpc` URLs, including leading `//rpc` and encoded exact aliases, return structured `404` errors. |
| `/` | GET | Dashboard UI. |

## Environment variables

| Variable | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA filtering, and a HIP-compatible ROCm overlay when `HIP_VISIBLE_DEVICES` is unset. Set it before starting KeepGPU; `--gpu-ids` selects visible ordinals after filtering, and service mode uses the daemon process environment. CUDA utilization telemetry maps visible ordinals through numeric tokens, full UUID tokens, or unique UUID prefixes before querying NVML; parsing stops at `-1` after any valid preceding tokens. Malformed, duplicate/equivalent, ambiguous, out-of-range, or unresolved mappings report unavailable telemetry instead of guessing a physical GPU. ROCm utilization treats this as one overlay on top of `ROCR_VISIBLE_DEVICES` and reports unavailable telemetry when it conflicts with `HIP_VISIBLE_DEVICES`. |
| `ROCR_VISIBLE_DEVICES` | ROCm base visibility mask. KeepGPU keeps `gpu_ids` as visible ordinals while resolving this mask before querying ROCm SMI telemetry. Unsupported, malformed, duplicate, out-of-range, or monitor-count-unverifiable ASCII numeric masks report unavailable utilization instead of guessing a physical SMI index. |
| `HIP_VISIBLE_DEVICES` | ROCm HIP-layer visibility overlay. If both `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set, they must describe the same ASCII numeric overlay for ROCm telemetry to query ROCm SMI; otherwise utilization is unavailable and non-negative `busy_threshold` values sleep for that cycle. |
| `CONSOLE_LOG_LEVEL` | Console log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `no`). |
| `FILE_LOG_LEVEL` | File log level; writes logs under `./logs/` when enabled. |
