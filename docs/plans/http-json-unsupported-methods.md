# HTTP JSON Unsupported Methods Plan

## Background

The HTTP service has explicit handlers for `GET`, `POST`, and `DELETE`. Methods
without a `do_*` handler, such as `HEAD`, `OPTIONS`, `PUT`, and `PATCH`, fall
through to `BaseHTTPRequestHandler.send_error()` and return an HTML `501`
response. That breaks the structured JSON contract for the REST API and
JSON-RPC surfaces.

## Goal

Return machine-readable JSON errors for unsupported HTTP methods on KeepGPU
API/RPC routes without changing existing supported `GET`, `POST`, or `DELETE`
behavior, and without changing the adjacent `GET /rpc` dashboard/static
fallback behavior.

## Solution

- Add HTTP regression tests for unsupported known API routes, `/rpc`, unknown
  `/api/*` routes, and `HEAD /api/sessions`.
- Intercept unsupported-method `501` dispatches only for known API/RPC surfaces.
- Return REST-shaped JSON `405 Method Not Allowed` responses with an `Allow`
  header for known API/RPC routes.
- Return REST-shaped JSON `404 Unknown endpoint` responses for unknown `/api/*`
  routes.
- Preserve HTTP `HEAD` semantics by sending headers for the JSON error payload
  without writing a response body.
- Document the invariant in `AGENTS.md`.

## Tasks

- [x] Add RED regression tests in `tests/mcp/test_http_api.py`.
- [x] Run the RED shard and confirm it fails because unsupported methods return
      HTML-backed `501` responses.
- [x] Implement narrow unsupported-method JSON handling in
      `src/keep_gpu/mcp/server.py`.
- [x] Update `AGENTS.md` with the API/RPC unsupported-method invariant.
- [x] Run targeted and broad verification commands.
- [x] Commit with `fix(mcp): return json for unsupported http methods`.

## Verification Log

- Baseline from coordinator:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` -> `57 passed`.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_api_known_routes_reject_unsupported_methods_with_json_405 tests/mcp/test_http_api.py::test_http_rpc_rejects_unsupported_options_with_json_405 tests/mcp/test_http_api.py::test_http_unknown_api_routes_reject_unsupported_methods_with_json_404 tests/mcp/test_http_api.py::test_http_head_api_sessions_rejects_with_json_405_headers_and_empty_body -q`
  failed with 7 failures. Each case returned HTTP `501` instead of the expected
  structured JSON `405` or `404`.
- GREEN focused shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_api_known_routes_reject_unsupported_methods_with_json_405 tests/mcp/test_http_api.py::test_http_rpc_rejects_unsupported_options_with_json_405 tests/mcp/test_http_api.py::test_http_unknown_api_routes_reject_unsupported_methods_with_json_404 tests/mcp/test_http_api.py::test_http_head_api_sessions_rejects_with_json_405_headers_and_empty_body -q`
  passed with 7 tests.
- HTTP API shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with
  64 tests.
- MCP/HTTP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  passed with 165 tests.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 538 tests and 11 skipped.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build` passed. It emitted the existing Material
  for MkDocs warning and unnav'd docs notices, including this plan page.
- Hooks:
  `pre-commit run --all-files` passed.
- Whitespace:
  `git diff --check` passed.

## Review Notes

- Local spec review found no must-fix issues.
- Local code-quality review found that `/rpc` must advertise the preserved
  `GET` fallback in `Allow`, and that multi-segment `/api/sessions/*` paths
  must not be classified as known session item routes for unsupported methods.
  Regression tests were added for both cases before the classifier fix.
- Review regressions:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_rpc_rejects_unsupported_options_with_json_405 tests/mcp/test_http_api.py::test_http_multisegment_session_route_rejects_unsupported_method_with_json_404 -q`
  failed before the fix with `/rpc` advertising only `POST` and
  `/api/sessions/foo/bar` returning `405`.
- Review regression green:
  the same command passed with 2 tests after the classifier fix.
- Hosted review:
  Gemini suggested guarding `self.path` and `self.command` before intercepting
  base-class unsupported-method errors. A focused regression reproduced the
  missing-attribute failure on an uninitialized handler, then passed after the
  guard was added.
