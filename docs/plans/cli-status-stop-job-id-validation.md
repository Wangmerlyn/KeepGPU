# CLI Status/Stop Job ID Validation Plan

## Background

`keep-gpu start` already validates custom `--job-id` values locally, but
`keep-gpu status --job-id` and `keep-gpu stop --job-id` passed explicit values
directly into the JSON-RPC service client. Empty, whitespace-only, or
non-URL-path-safe values therefore crossed the service boundary and could
produce service errors after unnecessary RPC attempts.

## Goal

Use the shared public `job_id` validator for CLI `status` and `stop` before any
service RPC, stop-all fallback, or daemon side effect. Omitted `--job-id`
continues to mean all-session status for `status`, and `stop` still requires
either `--job-id` or `--all`.

## Solution

- Add RED CLI tests proving invalid explicit `--job-id` values for `status` and
  `stop` do not call `_rpc_call()` or `_stop_all_sessions_with_fallback()`.
- Add one CLI helper around `session_config.validate_job_id()` and reuse it from
  `start`, `status`, and `stop`.
- Keep service-command failures as structured JSON error objects.
- Update CLI documentation and `AGENTS.md` with the local validation contract.

## Tasks

- [x] Add RED CLI regression tests for invalid `status`/`stop --job-id` values.
- [x] Implement shared CLI job-id validation before service calls.
- [x] Update `AGENTS.md`, README, CLI guide, reference docs, and this plan.
- [x] Run targeted tests, full tests, docs build, and pre-commit.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- Baseline before edits:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with `74 passed`.
- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'invalid_job_id_before'`
  failed with six cases because the commands still reached the monkeypatched RPC
  path instead of emitting a local JSON validation error.
- GREEN focused regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'invalid_job_id_before'`
  passed with `6 passed, 61 deselected`.
- Review follow-up:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'invalid_job_id_before or status_job_outputs'`
  passed with `7 passed, 61 deselected` after adding valid `status --job-id`
  forwarding coverage.
- GREEN CLI service-command file:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed
  with `68 passed`.
- GREEN CLI service-command shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with `78 passed`.
- Full branch gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `386 passed, 11 skipped`;
  `PYTHONPATH=$PWD/src mkdocs build` passed with the repository's existing
  Material warning and unnav'd plan notices; `pre-commit run --all-files`
  passed; and `git diff --check` passed.
- Local subagent code review: passed with no critical or important findings.
