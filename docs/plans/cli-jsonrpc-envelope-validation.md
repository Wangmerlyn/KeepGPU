# CLI JSON-RPC Envelope Validation Plan

## Background

`keep-gpu status`, `keep-gpu stop`, and `keep-gpu list-gpus` rely on
`_rpc_call()` for local service requests. `_rpc_call()` checks for an `"error"`
field, but otherwise returns `response.get("result", {})`. A malformed success
envelope that omits `"result"` can therefore be treated as an empty successful
response, which hides service/proxy bugs from users and downstream automation.

## Goal

Reject malformed JSON-RPC service envelopes as structured CLI errors instead of
accepting them as successful empty results or leaking tracebacks.

## Solution

- Add RED CLI tests proving malformed service envelopes exit nonzero
  and print a single-decode `{"error": "..."}` JSON object.
- Add focused `_rpc_call()` coverage for malformed response shapes.
- Validate the local service JSON-RPC envelope at the client boundary before
  returning a result.
- Update `AGENTS.md` and CLI docs so future CLI service changes keep the
  response-envelope contract.

## Tasks

- [x] Create an isolated worktree branch from latest `main`.
- [x] Run the focused CLI service-command baseline.
- [x] Add RED malformed-envelope tests.
- [x] Implement minimal `_rpc_call()` response-envelope validation.
- [x] Update `AGENTS.md`, docs, and this plan.
- [x] Run targeted tests, broader tests, docs build, and pre-commit.
- [ ] Run local subagent review.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

Completed so far:

- Baseline:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`,
  `41 passed`.
- RED missing-result regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_without_result tests/test_cli_service_commands.py::test_status_outputs_json_error_for_malformed_rpc_success_envelope -q`,
  `2 failed` because `_rpc_call()` did not raise and `keep-gpu status` exited
  successfully.
- GREEN missing-result regression:
  the same targeted command passed with `2 passed`.
- RED non-object result regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_non_object_result -q`,
  `1 failed` because list results were accepted.
- RED `jsonrpc`/`id` regressions:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_invalid_jsonrpc_version tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_mismatched_id -q`,
  `3 failed` because wrong or missing `jsonrpc` and mismatched `id` were
  accepted.
- RED non-object error regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_error_envelope_with_non_object_error -q`,
  `1 failed` because a malformed error envelope leaked as `AttributeError`.
- GREEN malformed-envelope group:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_without_result tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_non_object_result tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_invalid_jsonrpc_version tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_mismatched_id tests/test_cli_service_commands.py::test_rpc_call_rejects_error_envelope_with_non_object_error tests/test_cli_service_commands.py::test_status_outputs_json_error_for_malformed_rpc_success_envelope -q`,
  `7 passed`.
- Local review follow-up RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_non_object_jsonrpc_response tests/test_cli_service_commands.py::test_service_json_commands_output_json_error_for_non_object_rpc_response -q`,
  `4 failed` because a top-level JSON array leaked as `AttributeError` and
  `status`/`stop`/`list-gpus` printed no JSON error object.
- Local review follow-up GREEN:
  the same command passed with `4 passed` after adding a top-level response
  object guard. The expanded malformed-envelope group passed with `11 passed`.
- Gemini review follow-up RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_propagates_error_envelope_with_null_id tests/test_cli_service_commands.py::test_rpc_call_rejects_envelope_with_both_error_and_result -q`,
  `2 failed` because protocol-level errors with `id: null` were masked as
  mismatched ids and envelopes containing both `error` and `result` propagated
  as application errors.
- Gemini review follow-up GREEN:
  the same command passed with `2 passed` after checking mutual exclusion before
  error/result handling and allowing `id: null` for JSON-RPC error envelopes.
  The expanded malformed-envelope group passed with `13 passed`.
- Local review after Gemini follow-up RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_error_envelope_without_id_member -q`,
  `1 failed` because an omitted error-envelope `id` was treated the same as an
  explicit `id: null`.
- Local review after Gemini follow-up GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_error_envelope_without_id_member tests/test_cli_service_commands.py::test_rpc_call_propagates_error_envelope_with_null_id tests/test_cli_service_commands.py::test_rpc_call_rejects_envelope_with_both_error_and_result tests/test_cli_service_commands.py::test_rpc_call_rejects_success_envelope_with_mismatched_id -q`,
  `4 passed` after requiring the `id` member on error envelopes.
- GREEN CLI service-command shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`,
  `55 passed`.
- Targeted CLI + MCP/service shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/mcp/test_server.py -q`,
  `127 passed`.
- Full test suite:
  `PYTHONPATH=$PWD/src pytest tests -q`, `335 passed, 11 skipped`.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build`, passed with the known Material/MkDocs
  warning and unnav'd plan notices.
- Pre-commit:
  `pre-commit run --all-files`, passed.
