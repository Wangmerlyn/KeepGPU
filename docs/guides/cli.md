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
`job_id`.

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

### List telemetry

```bash
keep-gpu list-gpus
```

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
- active keep sessions,
- session creation form,
- single-session and stop-all controls.

## Command knobs

| Option | Meaning | Default |
| --- | --- | --- |
| `--gpu-ids` | Comma-separated GPU IDs. Omit to use all visible devices. | all |
| `--vram` | Per-GPU memory target (`512MB`, `1GiB`, bytes). | `1GiB` |
| `--interval` | Seconds between keep-alive cycles. | `300` |
| `--busy-threshold` / `--util-threshold` | Back off when utilization exceeds this value. | `-1` |

## Remote sessions

Preferred workflow for remote shells:

```bash
tmux new -s keepgpu
keep-gpu start --gpu-ids 0 --vram 1GiB --interval 120 --busy-threshold 25
```

Then run follow-up commands in the same shell (non-blocking), or monitor by way
of `keep-gpu status`.

## Troubleshooting

- **`--gpu-ids` parse error**: use only comma-separated integers (`0,1`).
- **Start cannot reach service**: run `keep-gpu serve --host 127.0.0.1 --port 8765`.
- **OOM during keep**: reduce `--vram` or free GPU memory before starting.
- **No utilization data**: ensure `nvidia-ml-py` works and `nvidia-smi` is available.
