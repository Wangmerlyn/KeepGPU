# HTTP Reject Negative Content-Length Plan

## Background

HTTP `POST` handling reads JSON request bodies through `_read_json_body()`.
Before this fix, the helper converted `Content-Length` with `int()` and then
passed the value directly to `self.rfile.read()`. `Content-Length: -1` therefore
became `read(-1)`, which can wait until the client closes the socket and tie up
a server thread instead of returning a parseable protocol error.

## Solution

Validate `Content-Length` as a non-negative integer before any body read.
Negative or non-integer values should use the existing parse-failure routing:
REST `/api/sessions` returns HTTP 400 with a structured JSON error, while
JSON-RPC `/` and `/rpc` return a JSON-RPC parse-error envelope with
`jsonrpc: "2.0"`, `id: null`, and `error.code == -32700`. Keep the existing
oversized-body check and message after the length is proven non-negative.

## Tasks

- [x] Add raw-socket regression tests that keep the client socket open and
      prove negative `Content-Length` receives a quick response.
- [x] Run the focused negative `Content-Length` tests before implementation and
      confirm they fail by timing out.
- [x] Add minimal `_read_json_body()` validation for negative and non-integer
      `Content-Length` values.
- [x] Update `AGENTS.md` with a concise body-length validation guideline.
- [x] Run focused and MCP test suites plus `git diff --check`.
- [x] Commit with `fix(mcp): reject invalid content length`.

## Verification Notes

- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_sessions_rejects_negative_content_length_without_client_close tests/mcp/test_http_api.py::test_http_jsonrpc_rejects_negative_content_length_with_parse_error -q`
  failed with 3 timeout assertions, proving negative `Content-Length` waited for
  client close before the fix.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_post_sessions_rejects_negative_content_length_without_client_close tests/mcp/test_http_api.py::test_http_jsonrpc_rejects_negative_content_length_with_parse_error -q`
  passed with 3 tests.
- Final targeted suite:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with 56
  tests.
- Broader MCP suite: `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with 147
  tests.
- Hygiene: `git diff --check` passed.
