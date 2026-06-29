# List GPUs Server Contract Plan

## Goal

Reject malformed GPU records at the server boundary before REST, direct JSON-RPC, or MCP responses can advertise unusable GPU targets.

## Scope

- Keep the fix local to server-side `list_gpus` behavior.
- Treat malformed records from `get_gpu_info()` as internal service failures.
- Preserve valid GPU records, including nullable memory and utilization fields.
- Avoid moving CLI validation helpers in this branch.

## Tasks

- [x] Add failing tests for malformed server `list_gpus` records over direct JSON-RPC, MCP `tools/call`, and `GET /api/gpus`.
- [x] Add a small server-side GPU record validator in `src/keep_gpu/mcp/server.py`.
- [x] Call the validator from `KeepGPUServer.list_gpus()` before returning public payloads.
- [x] Update `AGENTS.md` and user docs for the server-side list-gpus contract.
- [x] Run targeted tests, nearby tests, docs build, and pre-commit.
- [x] Address local review finding that `id` and `visible_id` must match.
- [ ] Run full-suite verification, local review, PR checks, and merge flow.

## Verification

- Red: `PYTHONPATH=src pytest tests/mcp/test_server.py::test_jsonrpc_list_gpus_rejects_malformed_gpu_record tests/mcp/test_server.py::test_mcp_tools_call_list_gpus_rejects_malformed_gpu_record tests/mcp/test_http_api.py::test_http_get_api_gpus_malformed_record_returns_json_500 -q` failed because malformed records received success envelopes/HTTP 200.
- Green: `PYTHONPATH=src pytest tests/mcp/test_server.py::test_list_gpus_accepts_nullable_memory_and_utilization tests/mcp/test_server.py::test_jsonrpc_list_gpus_rejects_malformed_gpu_record tests/mcp/test_server.py::test_mcp_tools_call_list_gpus_rejects_malformed_gpu_record tests/mcp/test_http_api.py::test_http_get_api_gpus_malformed_record_returns_json_500 -q` passed.
- Nearby: `PYTHONPATH=src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q` passed.
- Nearby: `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py tests/test_cli_service_commands.py -k 'list_gpus or list-gpus' -q` passed.
- Docs/hooks: `PYTHONPATH=$PWD/src mkdocs build`, `pre-commit run --all-files`, and `git diff --check` passed.
