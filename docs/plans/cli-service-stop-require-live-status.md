# CLI Service Stop Live Status Plan

## Background

`keep-gpu service-stop` currently skips the status RPC when `_service_available()`
returns false, then still calls `_stop_service_process()` in non-force mode. That
can signal an ownership-verified auto-started daemon without first proving there
are no tracked keep sessions.

## Goal

Require non-force `service-stop` to reach the service and complete the active
session status check before it can signal the managed daemon.

## Solution

- Keep `service-stop --force` unchanged: it may call `_stop_service_process()`
  directly, subject to ownership verification.
- In non-force mode, fail fast when the service is unavailable and tell users to
  use `keep-gpu service-stop --force` for an unresponsive auto-started daemon.
- Preserve the reachable-service paths: reject active sessions, or stop sessions
  by RPC and then stop the managed daemon when no active jobs are tracked.

## Tasks

- [x] Add a failing regression test proving unavailable non-force
      `service-stop` does not call `_stop_service_process()`.
- [x] Run the focused regression test and confirm it fails on current behavior.
- [x] Add the minimal CLI guard for unavailable non-force service-stop.
- [x] Add the concise `AGENTS.md` safety guideline.
- [x] Run the focused and requested regression checks.
- [x] Commit the scoped change.

## Verification Notes

- RED: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_service_stop_requires_live_status_without_force -q`
- GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_service_stop_requires_live_status_without_force -q`
- Focused file: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
- Broader CLI checks: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
- Diff hygiene: `git diff --check`

## Verification Results

- RED focused test failed before implementation because non-force
  `service-stop` reached `_stop_service_process()` when the service was
  unavailable.
- GREEN focused test passed after adding the non-force live-service guard.
- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed.
- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q` passed.
- `git diff --check` passed.
