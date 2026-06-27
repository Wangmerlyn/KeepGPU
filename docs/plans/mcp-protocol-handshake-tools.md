# MCP Protocol Handshake and Tools Plan

## Background

KeepGPU exposes `keep-gpu-mcp-server` and documents MCP client configuration, but
the current JSON-RPC dispatcher only accepts direct custom method names:
`start_keep`, `stop_keep`, `status`, and `list_gpus`. Real MCP clients first
send `initialize`, then discover tools with `tools/list`, then call tools with
`tools/call`. Those protocol methods currently fail as unknown methods.

The service also supports legacy direct JSON-RPC and REST/dashboard endpoints.
This fix must preserve those interfaces while adding the standard MCP protocol
entry points.

## Goal

Make `keep-gpu-mcp-server` usable by standard MCP clients without adding a new
runtime dependency or changing the controller contract.

## Solution

- Keep `KeepGPUServer` as the single backend.
- Add small protocol helpers in `src/keep_gpu/mcp/server.py`:
  - `initialize` returns the negotiated protocol version, server info, and tool
    capabilities.
  - `notifications/initialized` is accepted as a no-response notification.
  - `tools/list` returns tool descriptors with JSON input schemas for the four
    KeepGPU actions.
  - `tools/call` validates the requested tool name and routes arguments into the
    existing methods.
- Preserve direct JSON-RPC methods for existing scripts and tests.
- Keep responses JSON-RPC-compatible by including `jsonrpc: "2.0"` for normal
  request responses.
- Keep stdio line output protocol-clean by returning no line for notifications.

## Files

- Modify `src/keep_gpu/mcp/server.py`
- Modify `tests/mcp/test_server.py`
- Modify `tests/mcp/test_http_api.py` if HTTP protocol coverage needs a transport
  assertion beyond the shared dispatcher tests.
- Modify `README.md` and `docs/guides/mcp.md`
- Modify `AGENTS.md` to document MCP protocol parity expectations.

## Tasks

- [x] Add failing protocol tests for `initialize`, `notifications/initialized`,
      `tools/list`, and `tools/call`.
- [x] Add failing coverage that unknown MCP tool names return a JSON-RPC
      invalid-params error.
- [x] Add failing stdio coverage that stdout contains only protocol JSON.
- [x] Implement minimal protocol helpers and route them through `_handle_request`.
- [x] Keep legacy direct JSON-RPC calls working.
- [x] Update README, MCP guide, and AGENTS.md.
- [x] Run targeted MCP tests, then broader Python/docs/pre-commit checks.
- [x] Request local subagent review before PR.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/mcp -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
