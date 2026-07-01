# HTTP Encoded Leading-Slash Routes Plan

## Background

The HTTP API/RPC route classifiers rejected raw double-slash aliases such as
`//api/...` and `//rpc`, but missed aliases whose first extra slash was percent
encoded. For example, `/%2Fapi/gpus` decoded to `//api/gpus` and fell through to
static-file protection as `403`, while unsupported methods such as
`OPTIONS /%2Frpc` could leak `BaseHTTPRequestHandler` HTML `501` responses.

## Goal

Keep malformed API/RPC-looking route spellings inside KeepGPU's structured JSON
error boundary. Encoded leading-slash API/RPC aliases should return JSON
`404 Unknown endpoint`, with no dashboard/static fallback, HTML errors, or
JSON-RPC dispatch.

## Solution

- Add regression coverage for `/%2Fapi/gpus` and `/%2Frpc` across GET,
  unsupported method, and pre-dispatch parse paths.
- Add route-detection candidates that include decoded leading-slash collapses
  such as `//api/gpus -> /api/gpus` and `//rpc -> /rpc`.
- Use those candidates only for noncanonical route detection; canonical dispatch
  still uses the actual parsed path.
- Document the encoded leading-slash examples in `AGENTS.md` and
  `docs/guides/mcp.md`.

## Verification

- RED:
  `PYTHONPATH=src pytest tests/mcp/test_http_api.py::test_http_rpc_noncanonical_get_returns_json_404_without_static_fallback tests/mcp/test_http_api.py::test_http_rpc_encoded_exact_alias_rejects_before_jsonrpc_parse tests/mcp/test_http_api.py::test_http_rpc_encoded_exact_alias_unsupported_method_returns_json_404 tests/mcp/test_http_api.py::test_http_encoded_api_routes_return_json_404_without_static_fallback tests/mcp/test_http_api.py::test_http_encoded_api_route_unsupported_method_returns_json_404 tests/mcp/test_http_api.py::test_http_get_api_gpus_noncanonical_route_returns_json_404_without_listing -q`
  failed with `403` static fallback or HTML `501` for encoded leading-slash
  paths.
- GREEN:
  the same command passed with 25 tests after route candidate normalization.

## Remaining Checks

- [x] Run the MCP HTTP/server slice.
- [x] Run the full test suite.
- [x] Run `mkdocs build --strict`.
- [x] Run `pre-commit run --all-files --show-diff-on-failure`.
- [x] Run local subagent code review before PR.
