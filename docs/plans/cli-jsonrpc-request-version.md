# CLI JSON-RPC Request Version Plan

## Background

`keep-gpu status`, `keep-gpu stop`, and `keep-gpu list-gpus` call the local
service through `_rpc_call()`. The client already rejects malformed JSON-RPC
responses whose `jsonrpc` member is not `"2.0"`, but outgoing requests omitted
the version and therefore used the server's legacy direct-call compatibility
path.

## Goal

Make the CLI service client use standard JSON-RPC 2.0 request envelopes while
preserving legacy omitted-version handling for direct local scripts.

## Solution

- Add a regression test that captures `_rpc_call()`'s outgoing payload.
- Include `jsonrpc: "2.0"` in the request payload.
- Document the client/server split in AGENTS and CLI docs.

## Todo

- [x] Verify the focused CLI/MCP baseline.
- [x] Add the failing request-version regression test.
- [x] Add the minimal `_rpc_call()` payload change.
- [x] Run targeted verification.
- [x] Run broader verification.
- [x] Request local subagent review before PR.

## Verification

- RED:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py::test_rpc_call_sends_explicit_jsonrpc_request_version -q`
  failed because the captured request payload did not include `jsonrpc`.
- GREEN:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py::test_rpc_call_sends_explicit_jsonrpc_request_version -q`,
  `1 passed`.
- Targeted CLI/MCP contract suite:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py tests/mcp/test_server.py::test_jsonrpc_rejects_explicit_invalid_request_version tests/mcp/test_server.py::test_jsonrpc_accepts_explicit_valid_request_version tests/mcp/test_server.py::test_jsonrpc_omitted_version_legacy_direct_call_still_works -q`,
  `234 passed`.
- Full test suite:
  `PYTHONPATH=src pytest tests -q`, `944 passed, 11 skipped`.
- Docs and formatting:
  `mkdocs build --strict` passed with the known Material for MkDocs warning;
  `pre-commit run --all-files --show-diff-on-failure` passed.
- Local subagent review:
  reviewer found no functional issues and flagged that this plan file must be
  tracked before PR.
