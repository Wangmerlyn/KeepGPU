# JSON-RPC GPU IDs Invalid Params Plan

## Background

Direct JSON-RPC and MCP tool calls validate the shape of `gpu_ids` before
controller startup, but explicit visible-ordinal overflow is currently detected
inside `GlobalGPUController`. That controller raises a plain `ValueError`, so
JSON-RPC direct calls report `-32603 Internal error` for a normal user input
mistake. MCP `tools/call` likewise treats the same selection mistake as an
unexpected protocol error instead of a tool-level input error.

## Goal

Classify explicit out-of-range visible `gpu_ids` as public invalid parameters
for service entry points while preserving internal-error classification for
arbitrary controller failures.

## Solution

- Add RED coverage for direct JSON-RPC `start_keep` with an out-of-range visible
  CUDA ordinal returning `-32602 Invalid params`.
- Add RED coverage for MCP `tools/call start_keep` returning a successful tool
  envelope with `isError=true` for the same public input error.
- Introduce a specific controller exception for invalid visible GPU selection
  and map only that exception to `SessionInputError` in service startup.
- Preserve the existing test that arbitrary controller `ValueError` remains
  `-32603 Internal error`.
- Update `AGENTS.md`, MCP guide docs, and this plan.

## Tasks

- [x] Add RED tests for direct JSON-RPC and MCP tool invalid visible ordinal
      selection.
- [x] Confirm RED tests fail with the current internal-error behavior.
- [x] Implement the minimal controller exception and service mapping.
- [x] Update `AGENTS.md`, MCP docs, and this plan.
- [x] Run focused tests, full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent code review before PR.
- [ ] Open PR, resolve all review comments/checks, squash merge, and clean the
      worktree.

## Verification

- RED and GREEN focused tests:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_jsonrpc_start_keep_out_of_range_gpu_ids_returns_invalid_params tests/mcp/test_server.py::test_mcp_tools_call_out_of_range_gpu_ids_returns_tool_error tests/mcp/test_server.py::test_jsonrpc_start_keep_runtime_value_error_remains_internal_error -q`
  first failed with direct JSON-RPC returning `-32603` and MCP `tools/call`
  returning a protocol internal-error envelope. After implementation, the
  focused shard including the existing arbitrary-controller-`ValueError`
  regression passed with `3 passed`.
- MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with `179 passed`.
- Full no-GPU-safe gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `621 passed, 11 skipped`.
- Docs and hygiene:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan notices; `pre-commit run --all-files` passed after Black
  formatted the new assertions; after rerunning focused and full tests,
  `pre-commit run --all-files` and `git diff --check` passed.
- Local subagent review:
  The reviewer found no critical or important issues, identified this plan's
  focused-test command mismatch as a minor reproducibility issue, and marked the
  branch ready to merge after this documentation fix.
