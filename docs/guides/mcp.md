# MCP and Service API

KeepGPU ships a local service that powers three interfaces:

- JSON-RPC (`keep-gpu-mcp-server` or `/rpc`)
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

## JSON-RPC quick example

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

Custom `job_id` values are unique across active and starting sessions. If a
duplicate arrives while the original start is still creating controller work,
the duplicate is rejected before another controller begins keep-alive work.

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
CUDA and ROCm devices include memory and utilization when the platform APIs are
available. Mac M series devices report best-effort MPS memory counters and use
`null` for unsupported fields such as utilization.
When utilization is unavailable and `busy_threshold` is non-negative, controllers
sleep instead of running keepalive compute; `busy_threshold=-1` is the explicit
unconditional mode.
Stop controls show timed-out or failed releases instead of claiming success when
the backend keeps a session visible for follow-up cleanup. Retained session cards
show `Releasing` or `Release failed` with the backend error detail when present.

## Remote and security notes

- Bind to loopback (`127.0.0.1`) by default.
- For remote access, tunnel over SSH instead of exposing public ports.

```bash
ssh -L 8765:localhost:8765 gpu-box.example.com
```

- If you must expose externally, front with your own auth and reverse proxy.
