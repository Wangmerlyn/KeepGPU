# CLI Service Result Job ID Validation Plan

## Background

CLI service commands already validate user-supplied `--job-id` values and
`start_keep` result IDs with the shared URL-path-safe session contract. The
method-specific validators for `status` and `stop_keep` only checked
service-returned job IDs as strings, so malformed strings such as `bad/id`
could be emitted as successful machine-readable CLI output.

## Goal

Reject malformed service-returned job IDs in CLI `status` and `stop` results as
clean `ServiceResponseError` JSON errors before rendering user-facing output.

## Solution

- Add RED cases for URL-unsafe string IDs in all-session `status`, single-job
  `status`, and `stop_keep` outcome/error records.
- Reuse the centralized `validate_job_id()` contract inside CLI
  method-specific result validators.
- Keep `_rpc_call()` limited to generic JSON-RPC envelope validation.
- Update CLI and agent guidance so future service-result validation keeps job
  IDs aligned with the public session contract.

## Tasks

- [x] Create an isolated worktree branch from latest `main`.
- [x] Add RED malformed service-returned job-id tests.
- [x] Implement minimal CLI result job-id validation.
- [x] Update `AGENTS.md`, CLI guide, and this plan.
- [x] Run targeted tests, broader CLI tests, docs build, and pre-commit.
- [x] Run local subagent review.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_status_rejects_malformed_active_job_entries tests/test_cli_service_commands.py::test_status_job_rejects_malformed_payloads tests/test_cli_service_commands.py::test_stop_job_rejects_malformed_job_id_lists_and_errors -q`,
  `5 failed, 13 passed` because URL-unsafe string IDs were accepted.
- GREEN:
  the same command passed with `18 passed` after `status` and `stop_keep`
  result validators reused `validate_job_id()`.
- Broader CLI shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`,
  `179 passed`.
- Full test suite:
  `PYTHONPATH=$PWD/src pytest -q`, `775 passed, 11 skipped`.
- Docs and formatting:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the known Material
  warning; `pre-commit run --all-files --show-diff-on-failure` passed; `git diff
  --check` passed.
