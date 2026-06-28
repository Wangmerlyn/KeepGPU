# JSON-RPC Invalid ID Null Plan

## Background

`src/keep_gpu/mcp/server.py` stores `payload["id"]` before validating whether the
value is a legal JSON-RPC response id. Invalid request ids such as `true`,
arrays, or objects are then echoed in error envelopes, which can make strict MCP
or JSON-RPC clients reject the error response itself.

## Goal

Return `id: null` for missing or invalid JSON-RPC request ids, while preserving
the existing behavior that valid string and integer ids are echoed in responses.

## Solution

- Add MCP server tests for invalid JSON-RPC id types.
- Keep valid integer and string id behavior unchanged.
- Preserve id-less notification handling: id-less `notifications/*` messages do
  not receive responses, and missing-id requests receive `id: null` errors.
- Update `_handle_request` so the response id is assigned only after the raw id
  passes JSON-RPC id validation.

## Tasks

- [x] Add a failing regression test for boolean, object, and array request ids.
- [x] Verify the new test fails against the current implementation.
- [x] Implement the minimal handler change.
- [x] Run focused MCP tests and broader validation.
- [ ] Request local subagent code review before opening the PR.
- [ ] Resolve hosted PR review comments before squash merge.

## Verification Log

- RED: `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_mcp_invalid_request_id_types_return_null_id -q`
  failed with all three invalid ids echoed in `resp["id"]`.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_mcp_invalid_request_id_types_return_null_id tests/mcp/test_server.py::test_jsonrpc_rejects_explicit_invalid_request_version tests/mcp/test_server.py::test_jsonrpc_accepts_explicit_valid_request_version tests/mcp/test_server.py::test_jsonrpc_omitted_version_legacy_direct_call_still_works tests/mcp/test_server.py::test_mcp_requests_require_id tests/mcp/test_server.py::test_mcp_notification_with_id_is_invalid_request -q`
  passed with `8 passed`.
- Focused MCP: `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`
  passed with `98 passed`.
- MCP plus HTTP API:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  passed with `155 passed`.
- Full suite: `PYTHONPATH=$PWD/src pytest tests -q` passed with `509 passed,
  11 skipped`.
- Docs: `PYTHONPATH=$PWD/src mkdocs build` completed successfully. It emitted
  the repository's existing unlisted-plan-page warnings and the upstream
  Material for MkDocs 2.0 advisory.
- Hygiene: `pre-commit run --all-files` passed; `git diff --check --cached`
  produced no output.
- Local review: spec and code-quality reviewers found no must-fix issues. The
  optional coverage suggestion to include `False`, `None`, and `1.5` invalid ids
  was applied before PR.
