# CLI Start Job ID Response Validation Plan

## Background

`keep-gpu start` calls `_rpc_call("start_keep", ...)`, which validates the
generic JSON-RPC envelope and ensures the top-level `result` is an object. The
CLI then indexes `result["job_id"]` directly, so a buggy or hostile service can
return `{}`, `{"job_id": 123}`, or `{"job_id": ""}` and make the CLI leak a
raw exception or print unusable guidance.

## Goal/Solution

Reject malformed `start_keep` success payloads before rendering the start
success text. Keep valid service starts unchanged, but convert missing,
non-string, or empty `job_id` values into a clean `ServiceResponseError`.

## Tasks

- [x] Confirm the provided worktree is isolated on
      `codex/cli-validate-start-job-id`.
- [x] Add RED CLI regressions for missing, non-string, and empty `job_id`
      values returned by `start_keep`.
- [x] Preserve valid start-output expectations for a normal string `job_id`.
- [x] Implement minimal `start_keep` result validation in `src/keep_gpu/cli.py`.
- [x] Run focused RED, then GREEN, then requested verification commands.
- [x] Commit with `fix(cli): validate start job id response`.

## Verification Notes/Results

- RED focused tests:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_start_command_rejects_malformed_job_id_result tests/test_cli_service_commands.py::test_start_prints_dashboard_and_stop_hints -q`,
  `3 failed, 1 passed`; malformed `start_keep` results leaked `KeyError` or
  exited successfully before validation.
- GREEN focused tests:
  same command, `4 passed`.
- Requested shard `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`:
  `119 passed`.
- Requested shard `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py -q`:
  `12 passed`.
- `git diff --check`: passed.
