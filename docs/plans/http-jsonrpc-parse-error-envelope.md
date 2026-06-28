# HTTP JSON-RPC Parse Error Envelope Plan

## Background

HTTP `POST /rpc` and `POST /` are JSON-RPC compatibility endpoints. Before this
fix, malformed JSON bodies were rejected by the shared HTTP body parser before
the handler distinguished REST from JSON-RPC routes, so JSON-RPC clients saw a
REST-shaped HTTP 400 body such as `{"error":{"message":"Bad request: ..."}}`.
That body has no `jsonrpc`, `id`, or numeric JSON-RPC error code.

## Solution

Keep REST `/api/sessions` parse failures as structured HTTP 400 errors, but map
parse failures on `/` and `/rpc` to a JSON-RPC parse-error envelope:
`{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"..."}}`.

## Tasks

- [x] Add a failing HTTP JSON-RPC regression test for malformed `/rpc` JSON.
- [x] Confirm the test fails before implementation.
- [x] Return `_jsonrpc_error(None, JSONRPC_PARSE_ERROR, ...)` for `/` and `/rpc`
      parse failures.
- [x] Document JSON-RPC parse-error envelopes in the MCP guide and POST handler
      docstring.
- [x] Update `AGENTS.md` with the protocol boundary guideline.
- [x] Run targeted tests and `git diff --check`.
- [x] Commit with `fix(mcp): return jsonrpc parse error envelopes`.

## Verification Notes

- RED: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_jsonrpc_parse_error_returns_jsonrpc_envelope -q` failed because the route returned HTTP 400.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_jsonrpc_parse_error_returns_jsonrpc_envelope -q` passed after routing `/` and `/rpc` parse failures through the JSON-RPC envelope path.
- Final targeted suite: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with 51 tests.
- Broader MCP suite: `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with 142 tests.
- Hygiene: `pre-commit run --all-files`, `PYTHONPATH=$PWD/src mkdocs build`, and `git diff --check` passed.
