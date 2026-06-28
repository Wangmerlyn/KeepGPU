# CLI Service Payload Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CLI service commands reject malformed method-specific JSON-RPC result payloads with clean errors before rendering output or stopping daemons.

**Architecture:** Keep `_rpc_call()` focused on generic JSON-RPC envelope validation. Add small CLI-side result validators for `status`, `stop_keep`, and `list_gpus`, and call them immediately after each method-specific result is received.

**Tech Stack:** Python, Typer CLI, pytest, mkdocs, pre-commit.

---

## Background

The CLI already validates JSON-RPC envelopes in `_rpc_call()`: response object shape, `jsonrpc`, `id`, `error`, and top-level `result` object. The bug is that command handlers trust method-specific result fields after that point. A version-skewed or malformed service can return `{}` and be treated as success, including in non-force `service-stop` before daemon termination.

## Solution

Add permissive validators that require only the fields each CLI command consumes:

- `status` without `--job-id`: `active_jobs` must be a list.
- `status --job-id`: `active` must be a bool and `job_id` must be a string.
- `stop_keep`: `stopped`, `timed_out`, and `failed` must be lists; `errors` must be a dict; optional `message` must be a string.
- `list_gpus`: `gpus` must be a list and each element must be a dict.

Extra fields remain allowed so newer servers can add data without breaking older CLIs.

## Tasks

- [x] Write failing regression tests in `tests/test_cli_service_commands.py` for malformed `status`, `stop`, `list-gpus`, and non-force `service-stop` payloads.
- [x] Run the new regression tests before production changes and record RED evidence below.
- [x] Add a durable CLI result-validation invariant to `AGENTS.md`.
- [x] Implement minimal validators in `src/keep_gpu/cli.py` and call them after `_rpc_call()` results are received.
- [x] Validate `_stop_all_sessions_with_fallback()` direct and synthesized `stop_keep` results.
- [x] Run targeted and broad verification commands and record GREEN evidence below.
- [x] Commit with `fix(cli): validate service result payloads`.

## Verification Log

### RED

- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'malformed_all_session_payloads or status_job_rejects_malformed_payloads or stop_job_rejects_malformed_payloads or stop_all_rejects_malformed_payload or list_gpus_rejects_malformed_payloads or service_stop_rejects_malformed_status_before_side_effects or service_stop_rejects_malformed_stop_keep_before_stopping_daemon or status_job_outputs_single_decoded_json_object or service_stop_requires_managed_pid'`
  - Result: `19 failed, 2 passed, 120 deselected`.
  - Expected failure mode: malformed method-specific result payloads exited successfully (`exit_code == 0`) instead of returning clean CLI errors.

### GREEN

- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  - Result: `141 passed in 1.29s`.
- `PYTHONPATH=$PWD/src pytest tests -q`
  - Result: `531 passed, 11 skipped in 34.09s`.
- `PYTHONPATH=$PWD/src mkdocs build`
  - Result: passed. Existing warnings noted for MkDocs Material advisory and unnav'ed `docs/plans/` pages.
- `pre-commit run --all-files`
  - Result: passed.
- `git diff --check`
  - Result: passed.

## Review Notes

- Local spec and code-quality reviewers found no must-fix issues.
- The optional stop-all fallback safety-test suggestion was applied before PR.
