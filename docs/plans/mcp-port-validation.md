# MCP Port Validation Plan

## Background

The MCP HTTP executable validates endpoint values before binding, but
`argparse` currently parses `--port` with `type=int`. Non-integer values such as
`true` therefore fail with argparse's raw conversion message instead of the
shared KeepGPU endpoint-validation contract.

## Goal

Route MCP HTTP `--port` values through the same shared validation used by other
public endpoint surfaces, including non-integer values.

## Solution

- Tighten the MCP executable endpoint test to assert the shared error message.
- Let argparse keep `--port` as raw input and validate it with
  `validate_endpoint()`.
- Update MCP docs and agent guidance so future parser changes preserve the
  shared endpoint contract.

## Checks

- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_http_main_rejects_invalid_endpoint_before_binding -q`
- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k endpoint`
- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/utilities/test_endpoint_validation.py -q`
- `PYTHONPATH=$PWD/src pytest -q`
- `PYTHONPATH=$PWD/src mkdocs build --strict`
- `pre-commit run --all-files --show-diff-on-failure`
