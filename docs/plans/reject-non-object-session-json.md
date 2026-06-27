# Reject Non-Object Session JSON Plan

## Background

The REST `POST /api/sessions` handler decodes JSON and then assumes the payload
is an object. Array bodies such as `[]` can reach `payload.items()` and fall
through to the defensive HTTP 500 path instead of returning a user-correctable
400 response.

## Goal

Reject non-object REST session creation bodies with a clear HTTP 400 before
field validation or session creation. Malformed input must not create, reserve,
or mutate keep session state.

## Solution

- Add RED HTTP API tests for non-object JSON values sent to `POST /api/sessions`.
- Add an explicit `dict` shape check in the REST route before computing unknown
  fields or building `safe_payload`.
- Document that REST session creation expects a JSON object body.
- Keep JSON-RPC compatibility behavior unchanged.

## Tasks

- [x] Add RED tests proving list/scalar REST bodies return HTTP 400 and leave no
      active jobs.
- [x] Implement the minimal REST payload type guard in
      `src/keep_gpu/mcp/server.py`.
- [x] Update `AGENTS.md`, MCP guide, API reference, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- Baseline: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`
  passed with 25 tests.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_rejects_non_object_json_without_creating_session -q`
  failed with list bodies returning HTTP 500 and scalars returning inconsistent
  400 messages.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`
  passed with 29 tests.
- `PYTHONPATH=$PWD/src pytest tests -q` passed with 244 tests and 11 skipped.
- `PYTHONPATH=$PWD/src mkdocs build` passed with existing Material/MkDocs and
  unlisted-plan warnings.
- `pre-commit run --all-files` passed.
- `git diff --check` passed.
- Local subagent review had no blocking findings and noted the
  `_read_json_body()` annotation should reflect any decoded JSON value; updated
  it from `Dict[str, Any]` to `Any`.
