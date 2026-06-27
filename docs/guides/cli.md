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

Local `start` inputs are validated before auto-starting the service. Invalid
`--vram`, `--job-id`, `--interval`, `--busy-threshold`, or `--gpu-ids` values
fail without creating daemon runtime files or issuing RPC.

### Check status

```bash
keep-gpu status
keep-gpu status --job-id <job_id>
```

### Stop sessions

```bash
keep-gpu stop --job-id <job_id>
keep-gpu stop --all
```

### Stop local daemon

```bash
keep-gpu service-stop
```

If sessions are still active, stop them first or use `--force`. Force mode skips
the active-session RPC check, but it still stops only an ownership-verified
daemon that KeepGPU auto-started.

### List telemetry

```bash
keep-gpu list-gpus
```

The `id` field in this output is the visible ordinal to pass to `--gpu-ids`.
`status`, `stop`, and `list-gpus` print JSON objects, including
`{"error": "..."}` for service/runtime errors after CLI parsing succeeds, that
can be parsed directly with `jq` or a single `json.loads()` call.
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

## Command knobs

| Option | Meaning | Default |
| --- | --- | --- |
| `--gpu-ids` | Comma-separated unique non-negative visible device ordinals. Omit to use all visible devices; startup fails if that resolves to none or if an explicit ordinal is out of range. | all |
| `--vram` | Per-GPU memory target (`512MB`, `1GiB`, or bare bytes). | `1GiB` |
| `--interval` | Finite positive seconds between keep-alive cycles. | `300` |
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

- **`--gpu-ids` parse error**: use only comma-separated visible ordinals (`0,1`).
- **Unexpected GPU selection**: set `CUDA_VISIBLE_DEVICES` before starting
  KeepGPU on CUDA, or `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` before
  starting KeepGPU on ROCm, then pass visible ordinals by way of `--gpu-ids`.
  KeepGPU does not rewrite visibility masks in blocking mode. In service mode,
  ordinals are interpreted in the already-running service process environment.
- **Start cannot reach service**: run `keep-gpu serve --host 127.0.0.1 --port 8765`.
- **Need to close background service**: run `keep-gpu stop --all` first, then `keep-gpu service-stop`. Use `keep-gpu service-stop --force` only for an unresponsive auto-started daemon; it still refuses to signal a PID that KeepGPU cannot verify as its own.
- **OOM during keep**: reduce `--vram` or free GPU memory before starting.
- **No utilization data**: on CUDA, ensure `nvidia-ml-py` works and `nvidia-smi` is available; on ROCm, check the optional `rocm-smi` extra and avoid conflicting `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` masks. ROCm telemetry resolves visible ranks through `ROCR_VISIBLE_DEVICES` plus one HIP/CUDA overlay and reports unavailable utilization rather than guessing when the mapping is malformed or ambiguous. Valid `busy_threshold` values are `-1` or `0..100`, and omitted CLI values default to `25`. With non-negative `busy_threshold`, KeepGPU sleeps when utilization is unavailable. On Mac M series, utilization is expected to be `null`, so use `--busy-threshold -1` only when you intentionally want unconditional keepalive compute.
