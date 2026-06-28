# Controller Startup Unavailable Errors Plan

## Background

`GlobalGPUController` currently raises broad Python exceptions for two expected
startup-unavailable cases: unsupported platforms raise `NotImplementedError`,
and zero resolved visible GPUs raises `ValueError`. `KeepGPUServer.start_keep()`
cleans up reserved starting state and re-raises those exceptions. Direct
JSON-RPC then maps them through the generic `_handle_request()` defensive catch
as `-32603 Internal error`, while REST `POST /api/sessions` returns HTTP 500.

Existing tests intentionally preserve arbitrary startup/runtime `ValueError`,
`TypeError`, and `RuntimeError` as internal errors. This fix must keep that
boundary: only expected hardware/platform unavailability is public
startup-unavailable state.

## Goal

Expected startup-unavailable conditions should be explicit and parseable:
direct JSON-RPC returns `-32000`, REST session creation returns HTTP 503, MCP
`tools/call` returns `result.isError=true`, and no failed start leaves an active
session behind. Unexpected runtime failures remain internal.

## Solution

- Add controller-domain exceptions in
  `src/keep_gpu/global_gpu_controller/global_gpu_controller.py`.
- Raise those exceptions for unsupported controller platforms and zero visible
  GPU resolution, while preserving compatibility with existing
  `NotImplementedError` and `ValueError` expectations through multiple
  inheritance.
- Add a service-domain `SessionStartupUnavailable` in
  `src/keep_gpu/mcp/server.py`.
- Wrap `ControllerStartupUnavailable` raised during `start_keep()` into
  `SessionStartupUnavailable`.
- Map `SessionStartupUnavailable` to direct JSON-RPC
  `JSONRPC_STARTUP_UNAVAILABLE = -32000` and REST HTTP 503 structured errors.
- Leave `SessionInputError` as JSON-RPC `-32602`/REST 400 and arbitrary
  exceptions as JSON-RPC `-32603`/REST 500.

## Tasks

- [x] Add direct JSON-RPC tests for custom `SessionStartupUnavailable`, real
  CPU/unsupported platform startup, real zero-visible-GPU startup, and
  continued arbitrary `ValueError` internal behavior.
- [x] Add MCP `tools/call` test for startup-unavailable tool error text and no
  active jobs.
- [x] Add REST `POST /api/sessions` HTTP 503 test for
  `SessionStartupUnavailable` and preserve existing 500 tests.
- [x] Run focused RED tests and record failure evidence.
- [x] Implement controller and service exception mapping.
- [x] Run focused GREEN tests and record pass evidence.
- [x] Update `AGENTS.md`, `docs/guides/mcp.md`, and `docs/reference/cli.md`.
- [x] Run required verification commands and record evidence.
- [x] Commit with `fix(mcp): classify startup unavailable errors`.

## RED Evidence

Command:

```bash
PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_jsonrpc_start_keep_startup_unavailable_returns_public_code tests/mcp/test_server.py::test_jsonrpc_start_keep_unsupported_platform_returns_public_code tests/mcp/test_server.py::test_jsonrpc_start_keep_runtime_value_error_remains_internal_error tests/mcp/test_server.py::test_mcp_tools_call_startup_unavailable_returns_tool_error tests/mcp/test_http_api.py::test_http_post_sessions_startup_unavailable_returns_json_503 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_value_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_type_error_returns_json_500 -q
```

Result: failed as expected before production changes: 3 failed, 4 passed. The
custom startup-unavailable JSON-RPC test returned `-32603` instead of `-32000`;
the real CPU/unsupported-platform JSON-RPC test returned `-32603` instead of
`-32000`; the REST startup-unavailable test returned HTTP 500 instead of 503.
The arbitrary `ValueError`/`TypeError` internal-error tests and MCP tool text
test passed.

## GREEN Evidence

Command:

```bash
PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_jsonrpc_start_keep_startup_unavailable_returns_public_code tests/mcp/test_server.py::test_jsonrpc_start_keep_unsupported_platform_returns_public_code tests/mcp/test_server.py::test_jsonrpc_start_keep_zero_visible_gpus_returns_public_code tests/mcp/test_server.py::test_jsonrpc_start_keep_runtime_value_error_remains_internal_error tests/mcp/test_server.py::test_mcp_tools_call_startup_unavailable_returns_tool_error tests/mcp/test_http_api.py::test_http_post_sessions_startup_unavailable_returns_json_503 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_value_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_type_error_returns_json_500 -q
```

Result: passed after implementation: 8 passed. Direct JSON-RPC startup
unavailability now returns `-32000` for custom service startup failures, real
unsupported platforms, and real zero-visible-GPU startup. REST startup
unavailability returns HTTP 503 with `type="SessionStartupUnavailable"`, MCP
`tools/call` exposes the text as a tool error, and arbitrary
`ValueError`/`TypeError` cases remain internal.

## Verification Commands

Run in `/root/new_test/KeepGPU/.worktrees/codex/controller-startup-unavailable-errors`:

```bash
PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py tests/global_controller -q
PYTHONPATH=$PWD/src pytest tests -q
PYTHONPATH=$PWD/src mkdocs build
pre-commit run --all-files
git diff --check
```

## Verification Evidence

- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py tests/global_controller -q`: 203 passed, 1 skipped.
- `PYTHONPATH=$PWD/src pytest tests -q`: 493 passed, 11 skipped.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. MkDocs reported the existing Material for MkDocs 2.0 warning and existing `docs/plans/*` nav warnings, including this plan file.
- `pre-commit run --all-files`: initially reformatted `tests/mcp/test_server.py` with Black; rerun passed all hooks.
- `git diff --check`: passed with no whitespace errors.
