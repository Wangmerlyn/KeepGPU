# MCP Tool Internal Errors Plan

## Background

Direct JSON-RPC keeps public validation and startup-unavailable failures separate
from unexpected server failures. MCP `tools/call`, however, currently catches
every exception from KeepGPU tool execution and returns a successful JSON-RPC
response with `result.isError=true`.

That hides arbitrary controller/runtime failures as tool-level content instead
of surfacing the JSON-RPC `-32603 Internal error` envelope documented for
unexpected server failures.

## Goal

Keep MCP tool envelopes honest: expected startup-unavailable failures remain MCP
tool errors (`result.isError=true`), while unexpected internal failures return
JSON-RPC `-32603` like direct JSON-RPC.

## Solution

- Add RED coverage for `tools/call` when a KeepGPU method raises an unexpected
  `RuntimeError`.
- Preserve existing `SessionStartupUnavailable` behavior as
  `result.isError=true`.
- Preserve existing public tool-input validation behavior as
  `result.isError=true`, while protocol shape errors such as unknown tools still
  return JSON-RPC `-32602`.
- Let other unexpected exceptions propagate to `_handle_request()`, which already
  converts them to JSON-RPC `-32603`.
- Update verification evidence in this plan; AGENTS/docs already state the
  target contract.

## Tasks

- [x] Write failing MCP `tools/call` internal-error regression test.
- [x] Implement the minimal exception split in `_mcp_call_tool()`.
- [x] Run focused MCP tests and broader verification.
- [x] Run local subagent review and resolve findings before PR.
- [ ] Open PR, resolve hosted comments/checks, squash merge, and clean the
      worktree/branches.

## Verification Log

- Baseline before edits:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  passed with 169 passed.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_mcp_tools_call_unexpected_failure_returns_jsonrpc_internal_error -q`
  failed because the unexpected controller failure returned a successful
  `result.isError=true` response.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_mcp_tools_call_unexpected_failure_returns_jsonrpc_internal_error tests/mcp/test_server.py::test_mcp_tools_call_startup_unavailable_returns_tool_error tests/mcp/test_server.py::test_mcp_tools_call_rejects_oversized_integer_vram_as_tool_error tests/mcp/test_server.py::test_mcp_tools_call_unknown_tool_returns_protocol_error -q`
  passed with 4 passed after preserving validation/tool errors and propagating
  unexpected failures.
- Focused MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  passed with 170 passed.
- Broader affected tests:
  `PYTHONPATH=$PWD/src pytest tests/mcp tests/utilities/test_gpu_info.py tests/global_controller -q`
  passed with 249 passed, 2 skipped.
- Docs/hooks/whitespace:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan notices; `pre-commit run --all-files` passed; `git diff
  --check` passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 575 passed, 11 skipped.
- Local subagent review found that `docs/guides/mcp.md` still implied every
  `tools/call` failure kept a successful protocol envelope. Updated the guide to
  distinguish normal results, validation failures, startup-unavailable tool
  errors, protocol shape errors, and unexpected internal JSON-RPC `-32603`
  failures.
