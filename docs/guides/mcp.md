# MCP and Service API

KeepGPU ships a local service that powers three interfaces:

- MCP over stdio (`keep-gpu-mcp-server`)
- JSON-RPC over HTTP (`/rpc`)
- REST API (`/api/*`)
- Dashboard UI (`/`)

This is the same backend used by `keep-gpu start/status/stop/list-gpus`.

## Start service

### Preferred

```bash
keep-gpu serve --host 127.0.0.1 --port 8765
```

### MCP executable

```bash
keep-gpu-mcp-server --mode http --host 127.0.0.1 --port 8765
```

## MCP protocol over stdio

MCP clients should start with `initialize`, then discover KeepGPU actions with
`tools/list`, then invoke an action with `tools/call`. KeepGPU exposes these
tool names:

- `start_keep`
- `stop_keep`
- `status`
- `list_gpus`

Minimal client config:

```yaml
servers:
  keepgpu:
    command: ["keep-gpu-mcp-server"]
    adapter: stdio
```

The stdio transport writes only JSON protocol messages to stdout. KeepGPU logs
and diagnostics go to stderr so MCP clients can parse stdout safely.

JSON-RPC request messages that include a `jsonrpc` version must use `"2.0"`;
id-less `notifications/*` messages remain silent and do not produce response
envelopes.

## HTTP JSON-RPC quick example

HTTP mode is KeepGPU's local JSON-RPC/REST/dashboard service. It accepts the
same JSON-RPC message shapes at `/rpc`, but it is not a Streamable HTTP MCP
endpoint.

Malformed HTTP JSON-RPC bodies return a JSON-RPC `-32700 Parse error` envelope
with `id: null`; REST routes keep REST-shaped JSON errors.

```bash
curl -X POST http://127.0.0.1:8765/rpc \
  -H "content-type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}'
```

## Legacy JSON-RPC quick example

Direct method calls remain available for scripts and older local integrations.

```bash
curl -X POST http://127.0.0.1:8765/rpc \
  -H "content-type: application/json" \
  -d '{"id":1,"method":"start_keep","params":{"gpu_ids":[0],"vram":"512MB","interval":60,"busy_threshold":20}}'
```

Methods:

- `start_keep(gpu_ids?, vram?, interval?, busy_threshold?, job_id?)`
- `stop_keep(job_id?)`
- `status(job_id?)`
- `list_gpus()`

Successful direct-method responses are KeepGPU JSON-RPC envelopes with
`jsonrpc: "2.0"`, the matching request `id`, and an object `result`.

For direct JSON-RPC calls, public validation failures and unknown parameters
return JSON-RPC `-32602 Invalid params`. Expected startup-unavailable
conditions, such as an unsupported controller platform or no usable visible
GPUs, return `-32000` with the startup message. Unexpected server failures use
`-32603 Internal error`.

MCP `tools/call` responses keep protocol envelopes successful for normal tool
results, public tool-input validation failures, and expected hardware/platform
startup-unavailable failures. Those tool-level failures return
`result.isError=true` with the message in tool content. Protocol shape errors,
such as unknown tools, still return JSON-RPC errors such as `-32602`, and
unexpected internal controller/runtime failures return JSON-RPC
`-32603 Internal error`.

For supported REST route/method calls, service errors stay parseable. Public
validation failures return JSON `400` responses, unknown API routes return JSON
`404`, expected startup-unavailable session creation returns JSON `503`, and
unexpected runtime failures return JSON `500` responses with an `error` object.

REST session creation accepts a JSON object body, not arrays or scalar values.
Omitting `gpu_ids` means all GPUs visible to the service process. Omitting
`busy_threshold` uses the eco-safe default `25`. Explicit GPU values are visible
device ordinals in that same process environment. Empty, duplicate, or
out-of-range lists are invalid, and startup returns an error if the resolved
selection contains zero devices. `interval` must be a finite positive number of
seconds, including fractional seconds, within the Python runtime wait limit;
`NaN`, infinities, and oversized values are rejected before session creation.
`vram` accepts human-readable sizes or bytes, but byte-equivalent requests
above 1 PiB are rejected as public validation errors.
Cheap local fields (`vram`, `interval`, `busy_threshold`, `job_id`, duplicate
custom `job_id`, and `gpu_ids` shape) are rejected before `/api/sessions` asks
the service to list visible GPUs, so bad requests do not spend telemetry work.
CUDA telemetry resolves `CUDA_VISIBLE_DEVICES` for the service process and
treats malformed, duplicate/equivalent, ambiguous, or out-of-range masks as
unavailable utilization rather than partially querying or guessing a physical
GPU.

Custom `job_id` values are unique across active and starting sessions. If a
duplicate arrives while the original start is still creating controller work,
the duplicate is rejected before another controller begins keep-alive work.
Only `null`/omitted means "generate an ID" for `start_keep` or "all sessions"
for `status` and `stop_keep`. Custom IDs must be non-empty strings containing
only letters, digits, `.`, `_`, `-`, or `~`; invalid IDs return an error before
session state changes.

Status calls show reserved jobs as `state="starting"` while controller startup
is still in progress. That includes both `status(job_id)` and the all-session
`status()` list, so agents do not mistake an in-progress start for no session.
If an already-started worker later reports a terminal runtime or allocation
failure, the retained session is refreshed to `state="runtime_failed"` with
`last_error`. It remains visible and stoppable. This is distinct from normal
busy-GPU or unavailable-telemetry backoff, where the controller defers
allocation and the session stays active.

`stop_keep` returns additive outcome fields:

```json
{"stopped": ["job-a"], "timed_out": [], "failed": [], "errors": {}}
```

If a release times out, the session remains visible in `status` with
`state="stopping"` until the background release finishes. If that background
release later succeeds, the session is removed; if it fails, the session remains
visible with `state="stop_failed"` and `last_error` describing what happened. A
job id only appears in `stopped` after cleanup has completed within the stop
request timeout.

If `stop_keep` arrives while a matching session is still starting, the service
waits for startup to settle before deciding whether the job exists. Stop-all
requests also wait for in-progress starts before taking their session snapshot.
For stop-all, starts that begin after that request's initial snapshot are not
stopped by that request.
Stop-all releases the sessions in its snapshot concurrently and aggregates
results in deterministic snapshot order using the same additive fields.

## REST quick examples

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/api/gpus
curl http://127.0.0.1:8765/api/sessions
```

`/api/gpus` returns start-compatible visible ordinals as `id`/`visible_id`.
Optional `physical_id` or `uuid` fields describe the underlying vendor device
only; clients should not send those metadata values as `gpu_ids`.
On CUDA, NVML records are exposed only when Torch CUDA can start the same
visible ordinal set, so NVML-only devices are omitted instead of advertised as
usable session targets.
On ROCm, `physical_id` is included only when KeepGPU can safely resolve
`ROCR_VISIBLE_DEVICES` and one matching HIP/CUDA overlay to a ROCm SMI index;
otherwise utilization is reported as unavailable rather than guessed.

Start and stop by way of REST:

```bash
curl -X POST http://127.0.0.1:8765/api/sessions \
  -H "content-type: application/json" \
  -d '{"gpu_ids":[0],"vram":"1GiB","interval":120,"busy_threshold":25}'

curl -X DELETE http://127.0.0.1:8765/api/sessions/<job_id>
curl -X DELETE http://127.0.0.1:8765/api/sessions
```

## Dashboard

Open:

```text
http://127.0.0.1:8765/
```

The dashboard provides live telemetry, tracked session state, and start/stop controls.
Telemetry cards show the visible ordinal to type into the start form before any
physical/vendor metadata.
CUDA and ROCm devices include memory and utilization when the platform APIs are
available. Mac M series devices report best-effort MPS memory counters and use
`null` for unsupported fields such as utilization.
Dashboard summary cards average only finite utilization readings and show `n/a`
when every visible reading is unavailable, so unknown telemetry is not presented
as idle. Per-GPU cards also omit the utilization fill for unavailable readings.
Session cards may show retained runtime failures when a started worker reaches a
terminal allocation/runtime error; those sessions can still be stopped.
Valid `busy_threshold` values are `-1` or `0..100`, and omitted API values
default to `25`. When utilization is unavailable and `busy_threshold` is
non-negative, controllers sleep instead of allocating keep tensors or running
keepalive compute;
`busy_threshold=-1` is the explicit unconditional mode.
Stop controls show timed-out or failed releases instead of claiming success when
the backend keeps a session visible for follow-up cleanup. Retained session cards
show `Releasing` or `Release failed` with the backend error detail when present.
When REST calls fail, the footer uses the structured backend `error.message`
instead of showing the raw JSON error payload.

## Remote and security notes

- Bind to loopback (`127.0.0.1`) by default.
- For remote access, tunnel over SSH instead of exposing public ports.

```bash
ssh -L 8765:localhost:8765 gpu-box.example.com
```

- If you must expose externally, front with your own auth and reverse proxy.
