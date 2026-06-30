# CLI JSON-RPC Response ID Type Plan

## Background

The CLI service client already rejects missing, null, and mismatched JSON-RPC
response IDs. It compared response IDs with Python equality, so invalid JSON-RPC
ID types such as `1000.0` or `true` could compare equal to integer request IDs
and be accepted.

## Goal

Reject JSON-RPC service responses unless the response ID both equals the request
ID and uses the same valid ID type.

## Tasks

- [x] Create an isolated worktree branch from latest `main`.
- [x] Add RED tests for float and boolean response IDs on success and error
      envelopes.
- [x] Add a minimal shared response-ID matcher for `_rpc_call()`.
- [x] Update `AGENTS.md`, CLI guide, and this plan.
- [x] Run targeted tests and broader checks.
- [ ] Run local subagent review.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED:
  the four invalid-id cases now covered by
  `test_rpc_call_rejects_envelope_with_invalid_id_type` failed in the initial
  two-test draft because invalid response ID types were accepted.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_rejects_envelope_with_invalid_id_type -q`
  passed with `4 passed` after `_rpc_call()` required matching response ID type
  and value.
