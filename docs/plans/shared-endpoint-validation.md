# Shared Endpoint Validation

## Background

The CLI service commands and MCP HTTP executable now enforce the same public
endpoint contract: `--host` must be a DNS hostname or IPv4 literal, and `--port`
must be an integer in `1..65535`. The implementations are near-copies in
`src/keep_gpu/cli.py` and `src/keep_gpu/mcp/server.py`, including duplicated
error strings and DNS/IP parsing logic.

## Goal

Keep the endpoint contract identical across CLI and MCP while making the code
smaller and easier to audit.

## Solution

- Add a pure `src/keep_gpu/utilities/endpoint_validation.py` helper with shared
  error constants and `ValueError`-raising host/port validators.
- Keep CLI and MCP adapters thin:
  - CLI wraps utility `ValueError`s as `typer.BadParameter`.
  - MCP keeps `argparse` behavior and calls `parser.error(str(exc))`.
- Preserve the current behavior difference where CLI JSON-output commands pass
  string ports into validation for structured JSON errors, while MCP still lets
  `argparse(type=int)` reject non-integer port tokens before utility validation.
- Add focused utility tests and keep existing CLI/MCP endpoint regression tests.

## Validation

- `PYTHONPATH=$PWD/src pytest tests/utilities/test_endpoint_validation.py -q`
- `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k "validate_cli_service_host or invalid_host or invalid_port or non_integer_port or service_stop"`
- `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q -k "endpoint"`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
