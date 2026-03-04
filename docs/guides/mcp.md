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

The dashboard provides live telemetry, active sessions, and start/stop controls.

## Remote and security notes

- Bind to loopback (`127.0.0.1`) by default.
- For remote access, tunnel over SSH instead of exposing public ports.

```bash
ssh -L 8765:localhost:8765 gpu-box.example.com
```

- If you must expose externally, front with your own auth and reverse proxy.
