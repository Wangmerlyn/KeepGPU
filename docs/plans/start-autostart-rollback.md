# Start Auto-Start Rollback Plan

## Background

`keep-gpu start` can auto-start the local service daemon before it sends the
`start_keep` JSON-RPC request. If that request then fails with the service's
expected startup-unavailable error, such as no usable visible GPUs or an
unsupported platform, no keep session exists but the just-created daemon keeps
running. `_rpc_call()` currently drops the JSON-RPC `error.code`, so the CLI
cannot distinguish expected startup-unavailable failures from arbitrary RPC
errors.

## Goal

Keep service mode low-power and tidy by stopping a daemon that this exact
`start` invocation auto-started when the requested session fails before
creation with startup-unavailable JSON-RPC code `-32000`.

## Solution

- Preserve the JSON-RPC error code on `ServiceRPCError`.
- Teach `keep-gpu start` to best-effort stop the just-created daemon only when
  all of these are true:
  - this command auto-started the service,
  - `start_keep` returned `-32000`, and
  - no successful session result was received.
- Keep other RPC failures conservative: do not stop an already-running daemon,
  and do not stop for malformed success payloads or unrelated RPC errors.
- Document the lifecycle invariant in CLI docs and `AGENTS.md`.

## Tasks

- [x] Add RED tests for preserving JSON-RPC error codes and rolling back
      startup-unavailable auto-starts.
- [x] Confirm RED tests fail for the current implementation.
- [x] Implement the minimal CLI error-code preservation and rollback logic.
- [x] Update `AGENTS.md`, CLI guide/reference, and this plan.
- [x] Run focused tests, full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent code review before PR.
- [ ] Open PR, resolve all review comments/checks, squash merge, and clean the
      worktree.

## Verification

- RED and GREEN tests:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_propagates_error_envelope_with_null_id tests/test_cli_service_commands.py::test_start_rolls_back_auto_started_service_on_startup_unavailable -q`
  first failed with the expected missing `.code` attribute and missing rollback
  stop call. After implementation, the expanded focused shard
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_rpc_call_propagates_error_envelope_with_null_id tests/test_cli_service_commands.py::test_start_rolls_back_auto_started_service_on_startup_unavailable tests/test_cli_service_commands.py::test_start_does_not_stop_auto_started_service_for_non_startup_rpc_error tests/test_cli_service_commands.py::test_start_does_not_stop_already_running_service_on_startup_unavailable tests/test_cli_service_commands.py::test_start_does_not_stop_auto_started_service_for_malformed_success_payload tests/test_cli_service_commands.py::test_start_rollback_stop_failure_preserves_startup_error -q`
  passed with `6 passed`.
- CLI service shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed with
  `153 passed`.
- Full no-GPU-safe gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `619 passed, 11 skipped`.
- Docs and hygiene:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan notices; `pre-commit run --all-files` passed; and
  `git diff --check` passed.
- Local subagent review:
  The reviewer found no critical, important, or minor issues and marked the
  branch ready to merge.
