# MCP Server

Expose KeepGPU as a minimal JSON-RPC server (MCP-style) so agents or remote
orchestrators can start/stop keep-alive jobs and inspect GPU state.

## When to use this

- You run KeepGPU from an agent (LangChain, custom orchestrator, etc.) instead of a shell.
- You want to keep GPUs alive on a remote box over TCP rather than stdio.
- You need a quick way to list GPU utilization/memory by way of the same interface.

## Quick start

=== "stdio (default)"
    ```bash
    keep-gpu-mcp-server
    ```
    Send one JSON request per line:
    ```bash
    echo '{"id":1,"method":"start_keep","params":{"gpu_ids":[0],"vram":"512MB","interval":60,"busy_threshold":20}}' | keep-gpu-mcp-server
    ```

=== "HTTP"
    ```bash
    keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
    curl -X POST http://127.0.0.1:8765/ \
      -H "content-type: application/json" \
      -d '{"id":1,"method":"status"}'
    ```

Supported methods:

- `start_keep(gpu_ids?, vram?, interval?, busy_threshold?, job_id?)`
- `stop_keep(job_id?)` (omit `job_id` to stop all)
- `status(job_id?)` (omit `job_id` to list active jobs)
- `list_gpus()` (detailed info by way of NVML/ROCm SMI/torch)

## Client configs (MCP-style)

=== "stdio adapter"
    ```yaml
    servers:
      keepgpu:
        description: "KeepGPU MCP server"
        command: ["keep-gpu-mcp-server"]
        adapter: stdio
    ```

=== "HTTP adapter"
    ```yaml
    servers:
      keepgpu:
        url: http://127.0.0.1:8765/
        adapter: http
    ```

## Remote/cluster usage

- Run on the GPU host:
  ```bash
  keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
  ```
- Point your client at the host:
  ```yaml
  servers:
    keepgpu:
      url: http://gpu-box.example.com:8765/
      adapter: http
  ```
- If the network is untrusted, tunnel instead of exposing the port:
  ```bash
  ssh -L 8765:localhost:8765 gpu-box.example.com
  ```
  Then use `http://127.0.0.1:8765/` in your MCP config. For multi-user clusters,
  consider fronting the service with your own auth/reverse-proxy.

## Responses you can expect

```json
{"id":1,"result":{"job_id":"<uuid>"}}                # start_keep
{"id":2,"result":{"stopped":["<uuid>"]}}            # stop_keep
{"id":3,"result":{"active":true,"job_id":"<uuid>","params":{"gpu_ids":[0]}}}
{"id":4,"result":{"active_jobs":[{"job_id":"<uuid>","params":{"gpu_ids":[0]}}]}}
{"id":5,"result":{"gpus":[{"id":0,"platform":"cuda","name":"A100","memory_total":...,"memory_used":...,"utilization":12}]}}
```
