# CLI PID Ownership Strictness Plan

## Background

`keep-gpu service-stop --force` relies on a structured PID record before sending
signals to a local daemon. The matcher requires `uid` and `start_time` keys, but
it previously accepted records where either value was `null` when the live
process probe also returned `None`. That makes an unknown identity compare equal
to another unknown identity, which is too weak for daemon ownership.

## Goal

Only signal a service daemon when the PID record and the current process both
have known matching identity components: command line, UID, and process start
identity. Unknown recorded or current UID/start-time values must clear the stale
record and avoid signaling the process.

## Solution

- Add RED tests for PID records created with unknown UID or unknown start-time
  values.
- Require non-`None` recorded and current UID/start-time values in
  `_record_matches_running_process()`.
- Update `AGENTS.md` with the ownership strictness rule.

## Tasks

- [x] Create an isolated `.worktrees/codex/cli-pid-ownership-strict` branch
  from latest `origin/main`.
- [x] Add RED ownership tests for unknown recorded PID identity components.
- [x] Implement the minimal matcher strictness fix.
- [x] Run targeted ownership tests.
- [x] Run broader CLI/full verification, docs build, and pre-commit.
- [x] Request local subagent review and resolve findings.
- [ ] Open a PR, resolve hosted review comments, wait for green checks, squash
  merge, and clean up the worktree/branches.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_service_process_rejects_unknown_recorded_identity_components tests/test_cli_service_commands.py::test_stop_service_process_rejects_unknown_current_identity_components -q`
  failed because records with `uid=None` or `start_time=None` still sent
  SIGTERM/SIGKILL.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_service_process_rejects_unknown_recorded_identity_components tests/test_cli_service_commands.py::test_stop_service_process_rejects_unknown_current_identity_components tests/test_cli_service_commands.py::test_stop_service_process_stops_matching_owned_daemon tests/test_cli_service_commands.py::test_stop_service_process_rechecks_ownership_before_sigkill tests/test_cli_service_commands.py::test_stop_service_process_confirms_sigkill_exit -q`
  passed with 7 passed.
- CLI service:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed with
  145 passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 594 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan-page notices.
- Local review:
  a local subagent review found no Critical or Important issues. The Minor test
  hardening suggestion was addressed by asserting unverifiable records are
  cleared in both new unknown-identity tests.
