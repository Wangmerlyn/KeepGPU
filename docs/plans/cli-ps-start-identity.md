# CLI Process Start Identity Fallback Plan

## Background

KeepGPU stores a structured PID record for auto-started service daemons before
later service-stop or cleanup paths send signals. The record includes UID and a
process start identity so a reused PID is not mistaken for a managed daemon.

UID lookup already falls back to `ps` when `/proc` is unavailable, but start
identity lookup returns `None` as soon as `/proc/<pid>/stat` is missing. On
non-`/proc` platforms, that makes KeepGPU record `start_time: null` and then
correctly refuse to signal the daemon later. The safety rule is sound; the
identity probe is too Linux-specific.

## Goal

Recover a stable process start identity on platforms without `/proc` while
preserving the invariant that KeepGPU never signals a daemon unless the stored
record and the current process both have known matching identity components.

## Solution

- Add RED tests for `ps`-based start identity recovery and guarded fallback
  failure.
- Add a narrow `ps -p <pid> -o lstart=` fallback after the existing `/proc`
  lookup path fails.
- Keep `None` on empty or failed fallback output so unverifiable daemons remain
  unsignaled.
- Require recorded/current UID values to be plain integers and recorded/current
  start identity values to be non-empty strings before ownership can match.
- Update agent and CLI docs with the platform fallback behavior.

## Tasks

- [x] Create an isolated `.worktrees/codex/cli-ps-start-identity` branch from
  latest `origin/main`.
- [x] Add RED tests for non-`/proc` start identity recovery.
- [x] Implement the minimal start identity fallback.
- [x] Update `AGENTS.md` and CLI docs without expanding the README.
- [x] Run targeted CLI tests, full tests, docs build, and pre-commit.
- [x] Request local subagent review and resolve findings.
- [ ] Open a PR, resolve hosted comments/checks, squash merge, and clean up.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'process_start_identity or ps_start_identity'`
  failed with 2 failures because `_process_start_identity()` returned `None`
  without trying `ps`, and the stop path refused to signal the managed daemon
  with a missing recorded start identity.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'process_start_identity or ps_start_identity'`
  passed with 5 passed.
- Review hardening RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_service_process_rejects_malformed_identity_record_components -q`
  failed with 5 failures because malformed identity values could compare equal
  and reach signal paths.
- Ownership GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'process_start_identity or ps_start_identity or malformed_identity_record_components or unknown_recorded_identity_components or unknown_current_identity_components or stops_matching_owned_daemon'`
  passed with 15 passed.
- CLI service:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed
  with 206 passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 856 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the existing
  Material for MkDocs warning.
- Hooks:
  `pre-commit run --all-files --show-diff-on-failure` and `git diff --check`
  passed.
- Local review:
  first pass found no must-fix issues and suggested stricter identity type
  checks; after implementing that hardening, a second local review found no
  must-fix issues.
