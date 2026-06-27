# CLI Local Validation Before Service Plan

## Background

`keep-gpu start` validates `interval`, `busy_threshold`, and `gpu_ids` before
crossing the service boundary, but it forwards `vram` and `job_id` to the
daemon without local validation. When the service is unavailable, invalid
`--vram` or `--job-id` can auto-start the daemon, create runtime files, or
report service-unavailable before the user sees the local input error.

## Goal

Reject invalid `keep-gpu start` local inputs before service auto-start or RPC,
so bad user input has no daemon side effects.

## Solution

- Add RED CLI tests proving invalid `--vram` and `--job-id` do not call
  `_ensure_service_running()` or `_rpc_call()`.
- Reuse the shared public validators for `vram` and `job_id` in the CLI start
  path before service startup.
- Document the no-side-effect validation contract for service-mode `start`.

## Tasks

- [x] Add RED CLI regression tests for invalid `--vram` and `--job-id`.
- [x] Implement local validation before `_ensure_service_running()`.
- [x] Update `AGENTS.md`, CLI docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'rejects_local_inputs_before_auto_start'`
  failed because both invalid cases called `_ensure_service_running()`.
- GREEN focused regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'rejects_local_inputs_before_auto_start'`,
  `2 passed`.
- GREEN CLI service-command shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`,
  `41 passed`.

Final branch gate before local review:

- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`:
  `41 passed` after implementation and again after local review follow-up.
- `PYTHONPATH=$PWD/src pytest tests -q`: `254 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  version warning and docs-nav notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.
- Local subagent code review: passed with no critical or important findings.
  Review follow-up added explicit valid-path `--job-id` forwarding coverage.
