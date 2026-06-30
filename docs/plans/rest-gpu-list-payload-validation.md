# REST GPU List Payload Validation Plan

`POST /api/sessions` prevalidates explicit `gpu_ids` by listing visible GPUs
before starting controller work. That preflight must share the same strict
`list_gpus()` contract as `/api/gpus`, direct JSON-RPC, and MCP tools.

## Goal

Reject malformed `list_gpus()` envelopes or records as structured REST `500`
internal failures before deriving allowed GPU IDs or starting a session.

## Problem

The REST start preflight used `server_ref.list_gpus().get("gpus", [])`. That
made two malformed service states look valid:

- `{"gpus": [{"id": 0, "name": "GPU 0"}]}` could still start a session because
  only `id` was checked.
- `{}` was treated as an empty valid list and returned `503 No usable visible
  GPUs are available`.

Both cases bypassed the stricter GPU record validator already used by
`KeepGPUServer.list_gpus()`.

## Plan

- [x] Add RED HTTP regressions proving malformed records and missing `gpus`
      payloads do not create controllers and should return structured `500`
      errors.
- [x] Add a shared `_validate_list_gpus_payload()` helper that requires an
      object with a `gpus` list and then validates each record with the existing
      `_validate_list_gpus_records()` contract.
- [x] Use that helper in REST session creation before deriving allowed visible
      IDs.
- [x] Update AGENTS and user docs so future changes preserve the distinction
      between valid empty listings (`503`) and malformed listings (`500`).

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_start_rejects_malformed_gpu_listing_before_startup -q`
  failed with `200` for a malformed record and `503` for a missing `gpus`
  payload.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_start_rejects_malformed_gpu_listing_before_startup -q`
  passed with `2 passed`.
- Focused HTTP contract shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_start_validates_gpu_ids_against_listed_visible_ids tests/mcp/test_http_api.py::test_http_start_rejects_malformed_gpu_listing_before_startup tests/mcp/test_http_api.py::test_http_start_reports_startup_unavailable_when_gpu_listing_unusable tests/mcp/test_http_api.py::test_http_get_api_gpus_malformed_record_returns_json_500 -q`
  passed with `8 passed`.
- Full HTTP API shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with
  `93 passed`.
- Annotation compatibility:
  `PYTHONPATH=$PWD/src pytest tests/test_annotation_compat.py -q` passed with
  `7 passed`.
- MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with `221 passed`.
- Full test suite:
  `PYTHONPATH=$PWD/src pytest -q` passed with `785 passed, 11 skipped`.
- Docs and formatting:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the known Material
  MkDocs 2 warning, and
  `pre-commit run --all-files --show-diff-on-failure` passed.
