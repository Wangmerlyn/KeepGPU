# REST JSON Errors for GET and DELETE Plan

## Background

The HTTP service already returned JSON error objects for most `POST` failures,
but `GET` and `DELETE` route handlers could raise out of `BaseHTTPRequestHandler`
and close the socket before writing a response. Local agents, scripts, and
dashboards then had to handle a transport failure instead of a normal JSON error.

## Goal

Keep supported REST route/method failures machine-readable by returning
structured JSON `500` responses for unexpected handler/runtime errors while
preserving existing JSON `400` validation responses and JSON `404`
unknown-endpoint responses.

## Solution

- Add RED HTTP tests proving `GET /api/gpus` and `DELETE /api/sessions`
  backend failures return JSON `500` error objects instead of disconnecting.
- Add RED setup regressions proving missing handler server wiring returns JSON
  `500` for `GET`, `POST`, and `DELETE`.
- Add POST runtime regressions proving unexpected startup `TypeError` and
  `ValueError` failures return JSON `500` rather than client-validation `400`.
- Add a GitHub-review regression proving unknown POST routes return JSON `404`
  before reading missing or malformed request bodies.
- Add a GitHub-review regression proving explicit REST `gpu_ids` are rejected
  when `/api/gpus` lists no visible IDs.
- Centralize defensive runtime error formatting in `_json_runtime_error()`.
- Wrap `GET`, `POST`, and `DELETE` server dispatch setup and route bodies in the
  defensive JSON response path without changing public validation behavior.
- Document the REST error contract in `AGENTS.md`, README, and service docs.

## Tasks

- [x] Add RED runtime-failure tests for `GET` and `DELETE`.
- [x] Implement shared JSON `500` handling for unexpected REST runtime errors.
- [x] Add RED setup-failure tests for `GET`, `POST`, and `DELETE`.
- [x] Move server dispatch setup inside each verb's defensive error guard.
- [x] Narrow the POST `400` path to request parsing, public validation, and
      explicit `SessionInputError` failures.
- [x] Move POST unknown-endpoint handling before body parsing.
- [x] Reject explicit REST `gpu_ids` when the current visible ID set is empty.
- [x] Update `AGENTS.md`, README, MCP guide, CLI reference, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

Completed so far:

- Baseline `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`:
  `29 passed`.
- RED runtime regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_get_api_gpus_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_delete_sessions_runtime_error_returns_json_500 -q`
  failed with `RemoteDisconnected` and server tracebacks.
- GREEN runtime regression after shared JSON `500` handling: same command,
  `2 passed`.
- RED setup regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_get_setup_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_delete_setup_runtime_error_returns_json_500 -q`
  failed with `RemoteDisconnected` while looking up `keepgpu_server` before the
  defensive block.
- GREEN focused regressions:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_get_api_gpus_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_delete_sessions_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_get_setup_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_delete_setup_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_setup_runtime_error_returns_json_500 -q`:
  `5 passed`.
- Local review RED POST runtime regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_sessions_runtime_type_error_returns_json_500 -q`
  failed because unexpected startup `TypeError` still returned HTTP `400`.
- Local review GREEN POST runtime regressions:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_sessions_runtime_type_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_value_error_returns_json_500 -q`:
  `2 passed`.
- GitHub review RED unknown-route regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_unknown_api_route_returns_json_404_before_body_parse -q`
  failed because unknown POST routes with missing or malformed bodies returned
  HTTP `400`.
- GitHub review GREEN focused shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_unknown_api_route_returns_json_404_before_body_parse tests/mcp/test_http_api.py::test_http_post_setup_runtime_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_type_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_sessions_runtime_value_error_returns_json_500 tests/mcp/test_http_api.py::test_http_post_rejects_non_object_json_without_creating_session tests/mcp/test_http_api.py::test_http_post_rejects_unknown_fields -q`:
  `10 passed`.
- GitHub review RED no-visible-ID regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_start_rejects_explicit_gpu_ids_when_no_visible_ids -q`
  failed because `gpu_ids=[0]` started despite an empty visible ID set.
- GitHub review GREEN GPU-ID focused shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_start_rejects_explicit_gpu_ids_when_no_visible_ids tests/mcp/test_http_api.py::test_http_session_lifecycle tests/mcp/test_http_api.py::test_http_session_start_defaults_to_eco_safe_busy_threshold tests/mcp/test_http_api.py::test_http_session_start_preserves_explicit_unconditional_busy_threshold tests/mcp/test_http_api.py::test_http_start_validates_gpu_ids_against_listed_visible_ids -q`:
  `5 passed`.
- HTTP API shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q`: `39 passed`.
- Targeted MCP/GPU-info shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp tests/utilities/test_gpu_info.py -q`:
  `126 passed, 1 skipped`.
- Full test suite: `PYTHONPATH=$PWD/src pytest tests -q`:
  `278 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  warning and unnav'd plan notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.
- Local subagent code review: initial reviews found POST runtime error
  misclassification and docs wording that overpromised unsupported HTTP methods;
  both were fixed and re-review reported no Critical or Important findings.
- GitHub review comments: moved POST unknown-endpoint handling before body
  parsing and added explicit `BLE001` suppressions for the intentional
  GET/POST/DELETE runtime-boundary catches. Explicit REST `gpu_ids` are now
  rejected when no visible IDs are listed.
