# RPC GET Method JSON Error Plan

## Background

The HTTP MCP service documents `/rpc` as a JSON-RPC compatibility endpoint for
`POST` requests. The method classifier currently lists `/rpc` as allowing both
`GET` and `POST`, while `do_GET()` has no dedicated `/rpc` branch. As a result,
`GET /rpc` can fall through to the dashboard/static fallback and return HTML.

## Goal

Make `GET /rpc` return a machine-readable JSON `405 Method Not Allowed` response
with `Allow: POST`, while preserving `POST /rpc` JSON-RPC behavior and `GET /`
dashboard/static behavior.

## Solution

- Add RED regression coverage for `GET /rpc`.
- Keep a guard proving `GET /` still serves dashboard/static content.
- Make `/rpc` POST-only in the allowed-method classifier.
- Route `GET /rpc` through the existing structured JSON `405` helper instead of
  the static fallback.
- Add an explicit durable invariant to `AGENTS.md` if the current guidance does
  not name the `/rpc` GET/static fallback case.

## Tasks

- [x] Add RED regression tests in `tests/mcp/test_http_api.py`.
- [x] Run the focused RED test and confirm it fails because `GET /rpc` returns
      dashboard/static content instead of JSON `405`.
- [x] Implement the narrow server fix in `src/keep_gpu/mcp/server.py`.
- [x] Update `AGENTS.md` with the explicit `/rpc` GET invariant.
- [x] Run focused GREEN tests.
- [x] Run targeted MCP HTTP/API tests.
- [x] Run `mkdocs build`, `pre-commit run --all-files`, and `git diff --check`.
- [x] Run full `pytest` if cheap enough; otherwise record why it was skipped.
- [x] Resolve local review feedback by adding explicit `HEAD /rpc` 405 coverage.

## Verification Log

- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_rpc_rejects_unsupported_options_with_json_405 tests/mcp/test_http_api.py::test_http_rpc_get_rejects_with_json_405_instead_of_static_fallback -q`
  failed with 2 failures. `OPTIONS /rpc` returned `Allow: GET, POST` instead of
  `Allow: POST`; `GET /rpc` returned HTTP `200` instead of JSON `405`.
- GREEN focused:
  the same command passed with 2 tests.
- HTTP API shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with 67
  tests, including `test_http_health_and_static_index` for the `GET /`
  dashboard/static guard.
- MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with 168 tests.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 561 tests and 11 skipped
  after the local review follow-up.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build` passed. It emitted the existing Material
  for MkDocs warning and unnav'd docs notices, including this plan page.
- Hooks:
  `pre-commit run --all-files` passed.
- Whitespace:
  `git diff --check` passed.
- Local review follow-up:
  added `test_http_rpc_head_rejects_with_json_405_and_empty_body` so `HEAD /rpc`
  stays JSON-routed with `Allow: POST` and an empty body.
