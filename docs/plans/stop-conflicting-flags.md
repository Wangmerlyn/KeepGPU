# Stop Conflicting Flags Plan

## Background

`keep-gpu stop --job-id X --all` currently follows the stop-all path because
the CLI checks `--all` before the single-session stop branch. That makes a typo
dangerous: a command that names one job can release every tracked keep session.

## Goal

Reject ambiguous stop targets before any RPC, stop-all fallback, or daemon stop
side effect.

## Solution

- Add a CLI regression test for `stop --job-id job-1 --all`.
- In `src/keep_gpu/cli.py`, check for both `job_id is not None` and
  `all_sessions` before the existing missing-target validation.
- Return the same structured JSON error shape used by other service commands.
- Document the mutual-exclusion rule in agent guidance and user CLI docs.

## Tasks

- [x] Add the RED regression test in `tests/test_cli_service_commands.py`.
- [x] Confirm the test fails on the existing stop-all behavior.
- [x] Implement the minimal early CLI validation guard.
- [x] Update `AGENTS.md`, `README.md`, `docs/reference/cli.md`, and
      `docs/guides/cli.md`.
- [x] Run targeted verification and record the results.

## Validation

- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_rejects_job_id_with_all_before_rpc -q`
  failed with `assert 0 == 1`, confirming the command still took the stop-all
  path.
- GREEN focused regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_rejects_job_id_with_all_before_rpc -q`
  passed with `1 passed in 0.15s`.
- CLI service-command shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with `61 passed in 1.07s`.
- Diff hygiene:
  `git diff --check` passed with no output.
