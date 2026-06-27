# JSON-RPC Invalid Params Code Plan

## Background

Direct legacy JSON-RPC calls to `start_keep`, `stop_keep`, and `status` use the
shared public validators, but their `ValueError`s currently fall through
`_handle_request()`'s generic exception handler and return `-32603 Internal
error`. These are user input errors and should return JSON-RPC `-32602 Invalid
params`.

## Goal

Map public parameter validation failures from direct JSON-RPC method calls to
`-32602` while preserving `-32603` for real internal failures.

## Solution

- Add RED assertions that direct JSON-RPC validation errors return `-32602`.
- Add RED coverage for invalid `vram` values and unknown direct-method
  parameters.
- Explicitly validate allowed params for direct KeepGPU methods before
  dispatch.
- Mark public session validation failures with a narrow `SessionInputError`
  wrapper so JSON-RPC maps user input errors to `-32602` without converting
  controller/runtime `ValueError`s away from `-32603`.
- Document the JSON-RPC error-code contract.

## Tasks

- [x] Add RED JSON-RPC error-code tests for existing validation cases.
- [x] Add RED invalid `vram` and unknown-parameter tests.
- [x] Add RED regression that runtime `ValueError`s still return `-32603`.
- [x] Implement explicit direct-method param validation and validation-error
      mapping.
- [x] Update `AGENTS.md`, MCP docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED focused regression run before implementation:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k 'jsonrpc_rejects_empty_gpu_ids or jsonrpc_rejects_duplicate_gpu_ids or jsonrpc_rejects_non_positive_interval or jsonrpc_rejects_nan_interval_without_creating_session or jsonrpc_rejects_busy_threshold_above_percent_range or jsonrpc_rejects_invalid_vram_type_without_creating_session or jsonrpc_rejects_unknown_direct_method_param_without_internal_error or jsonrpc_stop_keep_rejects_empty_job_id_without_stopping_sessions'`
  failed with `-32603 != -32602`.
- GREEN focused regression run after implementation: same command, `8 passed`.
- GREEN MCP server shard after implementation:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`, `70 passed`.
- Local subagent code review found the first implementation caught every
  `ValueError` from direct method dispatch, incorrectly mapping controller
  startup failures to `-32602`.
- RED review regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k 'runtime_value_error_remains_internal_error'`
  failed with `-32602 != -32603`.
- GREEN focused review regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k 'runtime_value_error_remains_internal_error or jsonrpc_rejects_invalid_vram_type_without_creating_session or jsonrpc_rejects_unknown_direct_method_param_without_internal_error or jsonrpc_rejects_empty_gpu_ids or jsonrpc_stop_keep_rejects_empty_job_id_without_stopping_sessions'`,
  `5 passed`.
- GREEN MCP server shard after review fix:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`, `71 passed`.

- Pre-review branch gate, to rerun after the review fix:
  - `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`: `70 passed`.
  - `PYTHONPATH=$PWD/src pytest tests -q`: `251 passed, 11 skipped`.
  - `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
    version warning and docs-nav notices were emitted.
  - `pre-commit run --all-files`: passed.
  - `git diff --check`: passed.

Post-review branch gate before local re-check:

- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`: `71 passed`.
- `PYTHONPATH=$PWD/src pytest tests -q`: `252 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  version warning and docs-nav notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.
- Local subagent code review re-check: passed with no critical, important, or
  minor findings.
