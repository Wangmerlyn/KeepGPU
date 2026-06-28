# REST Session Prevalidation Plan

## Background

`POST /api/sessions` validates `gpu_ids` shape before checking visible GPU IDs,
but when `gpu_ids` is present it calls `list_gpus()` before validating other
cheap public inputs such as `vram`, `interval`, `busy_threshold`, `job_id`, or
duplicate `job_id`. That means clearly invalid REST requests can still trigger
GPU telemetry probing before returning a user-correctable `400`.

## Goal

Reject invalid REST session creation inputs before telemetry or session state
work, while preserving visible-ID validation for otherwise valid explicit
`gpu_ids`.

## Solution

- Add RED HTTP tests proving invalid REST payload fields return JSON `400`
  without calling `list_gpus()`.
- Validate the full REST session payload with shared public validators before
  visible-GPU telemetry checks.
- Check existing/starting custom `job_id` values before visible-GPU telemetry,
  then keep the existing atomic duplicate reservation in `start_keep()`.
- Keep successful explicit `gpu_ids` behavior unchanged: valid IDs are still
  checked against `/api/gpus`-compatible visible IDs before starting.
- Document the validation-before-telemetry contract in `AGENTS.md`, MCP guide,
  and this plan.

## Tasks

- [x] Add RED tests for invalid `vram`, `job_id`, `interval`,
      `busy_threshold`, and duplicate `job_id` before `list_gpus()`.
- [x] Implement REST payload prevalidation before visible-GPU telemetry.
- [x] Preserve current valid explicit `gpu_ids` listing behavior.
- [x] Update `AGENTS.md`, MCP docs, and this plan.
- [x] Run focused MCP/HTTP tests, broader tests, docs build, and pre-commit.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge,
      and clean the worktree.

## Verification

Completed so far:

- Focused MCP/HTTP baseline before edits:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py tests/mcp/test_server.py -q`,
  `111 passed`.
- RED replay in a disposable worktree with tests only:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q -k 'invalid_fields_before_listing_gpus or duplicate_job_id_before_listing_gpus'`,
  `5 failed` because `list_gpus()` ran before cheap validation and returned
  HTTP `500` instead of `400`.
- GREEN focused regressions:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q -k 'invalid_fields_before_listing_gpus or duplicate_job_id_before_listing_gpus'`,
  `5 passed`.
- HTTP API shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`, `44 passed`.
- MCP/HTTP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py tests/mcp/test_server.py -q`,
  `116 passed`.
- Full test suite:
  `PYTHONPATH=$PWD/src pytest tests -q`, `317 passed, 11 skipped`.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build`, passed with the known
  Material/MkDocs warning and unnav'd plan notices.
- Pre-commit:
  `pre-commit run --all-files`, passed.
- Diff whitespace:
  `git diff --check`, passed.
- Local subagent review:
  spec compliance and code-quality reviewers reported no critical, important,
  or minor findings and marked the branch ready for PR.
