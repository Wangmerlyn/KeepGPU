# CLI Playbook

KeepGPU now supports two operational styles:

- **Blocking mode** (`keep-gpu ...`) for traditional shell workflows.
- **Service mode** (`keep-gpu start/status/stop`) for agent workflows that must continue after arming keep-alive.

## 1) Blocking mode (compatibility)

```bash
keep-gpu --interval 120 --gpu-ids 0,1 --vram 2GiB --busy-threshold 25
```

This command blocks until you press `Ctrl+C`.

## 2) Non-blocking service mode (recommended for agents)

### Start a keep session

```bash
keep-gpu start --gpu-ids 0 --vram 1GiB --interval 60 --busy-threshold 25
```

`start` auto-starts the local service if needed and returns immediately with a
`job_id`. The command also prints:

- dashboard URL (`http://<host>:<port>/`),
- follow-up status/stop command hints,
- daemon shutdown hint (`keep-gpu service-stop`).

If this invocation auto-starts the service but session creation fails with an
expected startup-unavailable error before a session is created, `start`
best-effort stops the just-created daemon so it does not remain idle.
If auto-start times out before the service becomes healthy, the CLI applies the
same best-effort cleanup to the daemon it just started.

Local `start` inputs are validated before auto-starting the service. Invalid
`--vram`, `--job-id`, `--interval`, `--busy-threshold`, or `--gpu-ids` values
fail without creating daemon runtime files or issuing RPC. Service endpoint
flags are validated the same way: `--host` must be a DNS hostname or IPv4
address, and `--port` must be an integer in `1..65535`.
Omit `--gpu-ids` to use all visible GPUs; an explicit empty or whitespace-only
value is invalid.
`--interval` must be finite, positive, and within the Python runtime wait
limit; fractional seconds such as `0.5` are accepted. `--vram` keeps integer
and digit-only values as bytes, accepts human units, and rejects
byte-equivalent requests above 1 PiB.

### Check status

```bash
keep-gpu status
keep-gpu status --job-id <job_id>
```

Explicit `--job-id` values use the shared session-id rules: non-empty strings
containing only letters, digits, `.`, `_`, `-`, or `~`. Invalid IDs return a
JSON error before contacting the service.

### Stop sessions

```bash
keep-gpu stop --job-id <job_id>
keep-gpu stop --all
```

Use exactly one stop target. `--job-id` and `--all` are mutually exclusive, and
passing both returns a JSON error before contacting the service or running the
stop-all fallback.
Explicit `--job-id` values use the same non-empty URL-path-safe validation as
`status`; invalid IDs return a JSON error before any RPC or stop-all fallback.

### Stop local daemon

```bash
keep-gpu service-stop
```

If sessions are still active, timed out, or failed to stop cleanly, resolve them
first or use `--force`. Force mode skips the session RPC checks, but it still
stops only an ownership-verified daemon that KeepGPU auto-started.

### List telemetry

```bash
keep-gpu list-gpus
```

The `id` field in this output is the visible ordinal to pass to `--gpu-ids`.
On CUDA, NVML telemetry is listed only when Torch CUDA can start the same
visible ordinal set; NVML-only devices are hidden rather than advertised as
usable selections. On ROCm, listed records are limited to visible ordinals that
Torch can select; nullable memory fields mean memory telemetry is unavailable
after selection succeeds. `status`, `stop`, and `list-gpus` print JSON objects,
including `{"error": "..."}` for service/runtime errors after CLI parsing
succeeds, that can be parsed directly with `jq` or a single `json.loads()` call.
Malformed JSON-RPC service envelopes and malformed method-specific result
records are reported as JSON error objects instead of empty success results.
Invalid service endpoints, including non-integer ports, are also reported as
JSON errors before any RPC or stop-all fallback runs.
When KeepGPU can identify the underlying device, it reports `physical_id` or
`uuid` as metadata; those metadata values are not accepted as `--gpu-ids`.

### Run service explicitly

```bash
keep-gpu serve --host 127.0.0.1 --port 8765
```

## 3) Dashboard UI

When service mode is running, open:

```text
http://127.0.0.1:8765/
```

The dashboard provides:

- live GPU memory/utilization telemetry,
- visible GPU ordinals to type into the start form,
- tracked keep sessions, including in-progress starts and releases,
- session creation form,
- single-session and stop-all controls.

Telemetry refresh is manual by default so an idle browser tab does not keep
probing GPU backends. Use **Refresh Now** for a one-shot update, or enable
**Auto refresh** for 10-second polling while the tab is visible.

Unavailable utilization readings are shown as `n/a`; they are excluded from
summary averages and do not draw an idle-looking utilization fill.

## Command knobs

| Option | Meaning | Default |
| --- | --- | --- |
| `--gpu-ids` | Comma-separated unique non-negative visible device ordinals. Omit to use all visible devices; explicit empty or whitespace-only values are invalid. Startup fails if all-visible resolution finds no GPUs or if an explicit ordinal is out of range. | all |
| `--vram` | Per-GPU memory target (`512MB`, `1GiB`, or bare bytes), capped at 1 PiB byte-equivalent. | `1GiB` |
| `--interval` | Finite positive seconds between keep-alive cycles, including fractional values, capped by the Python runtime wait limit. | `300` |
| `--busy-threshold` / `--util-threshold` | `0..100` backs off when utilization exceeds this value or telemetry is unavailable; `-1` disables utilization backoff. | `25` |
| `--job-id` | Optional URL-path-safe session id. Invalid IDs are rejected before service auto-start. | auto |

## Remote sessions

Preferred workflow for remote shells:

```bash
tmux new -s keepgpu
keep-gpu start --gpu-ids 0 --vram 1GiB --interval 120 --busy-threshold 25
```

Then run follow-up commands in the same shell (non-blocking), or monitor by way
of `keep-gpu status`.

## Troubleshooting

- **`--gpu-ids` parse error**: use only comma-separated visible ordinals (`0,1`). Omit the option to use all visible GPUs; do not pass an empty string.
- **Unexpected GPU selection**: set `CUDA_VISIBLE_DEVICES` before starting
  KeepGPU on CUDA, or `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` before
  starting KeepGPU on ROCm, then pass visible ordinals by way of `--gpu-ids`.
  KeepGPU does not rewrite visibility masks in blocking mode. In service mode,
  ordinals are interpreted in the already-running service process environment.
- **Start cannot reach service**: run `keep-gpu serve --host 127.0.0.1 --port 8765`.
- **Need to close background service**: run `keep-gpu stop --all` first, then `keep-gpu service-stop`. Use `keep-gpu service-stop --force` only for an unresponsive auto-started daemon; it still refuses to signal a PID that KeepGPU cannot verify as its own.
- **OOM during keep**: reduce `--vram` or free GPU memory before starting.
- **No utilization data**: on CUDA, ensure `nvidia-ml-py`/`pynvml` can initialize NVML; `nvidia-smi` can help sanity-check the driver outside KeepGPU. Malformed, duplicate/equivalent, or ambiguous `CUDA_VISIBLE_DEVICES` masks are reported as unavailable utilization rather than guessed. On ROCm, ensure the ROCm/system stack provides `rocm_smi` and avoid conflicting `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` masks; KeepGPU handles missing `rocm_smi` gracefully. ROCm telemetry resolves visible ranks through `ROCR_VISIBLE_DEVICES` plus one HIP/CUDA overlay and reports unavailable utilization rather than guessing when the mapping is malformed or ambiguous. Valid `busy_threshold` values are `-1` or `0..100`, and omitted CLI values default to `25`. With non-negative `busy_threshold`, KeepGPU sleeps before allocation or compute when utilization is unavailable. On Mac M series, utilization is expected to be `null`, so use `--busy-threshold -1` only when you intentionally want unconditional keepalive compute.
