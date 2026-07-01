# MCP list_gpus Unavailable Plan

## Background

`GET /api/gpus` already treats expected `DeviceEnumerationUnavailableError`
as a structured startup-unavailable response. Direct JSON-RPC `list_gpus` and
MCP `tools/call list_gpus` still route the same expected failure through the
generic internal-error path.

## Goal

Keep GPU enumeration failure semantics consistent across public service
surfaces: direct JSON-RPC should return `-32000`, and MCP tool calls should
return a successful protocol envelope with `result.isError=true`.

## Solution

- Add regression tests for direct JSON-RPC `list_gpus`.
- Add regression tests for MCP `tools/call list_gpus`.
- Classify `DeviceEnumerationUnavailableError` alongside other expected
  startup-unavailable service failures.
- Update service docs and agent guidance.

## Checks

- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k 'list_gpus or startup_unavailable or tools_call'`
- `PYTHONPATH=$PWD/src pytest tests/mcp tests/utilities/test_gpu_info.py -q`
- `PYTHONPATH=$PWD/src pytest -q`
- `PYTHONPATH=$PWD/src mkdocs build --strict`
- `pre-commit run --all-files --show-diff-on-failure`
