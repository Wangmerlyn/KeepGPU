# MCP Executable Endpoint Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for the code change. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `keep-gpu-mcp-server --mode http` and `python -m keep_gpu.mcp.server --mode http` reject invalid endpoint flags before attempting to bind a socket.

**Architecture:** Keep validation local to `src/keep_gpu/mcp/server.py` so the MCP executable does not import the Typer CLI module. Mirror the CLI service contract for small endpoint checks: host must be a DNS hostname or IPv4 address, and port must be an integer in `1..65535`.

**Tech Stack:** Python `argparse`, pytest, KeepGPU MCP server entrypoint.

---

## Background

`keep-gpu serve` validates `--host` and `--port` before starting the HTTP server. The direct MCP executable path currently parses `--port` as `int` but passes values straight to `run_http`. Audit repros show invalid values can surface as low-level socket exceptions:

- `PYTHONPATH=$PWD/src python -m keep_gpu.mcp.server --mode http --port 70000` raises `OverflowError`.
- `PYTHONPATH=$PWD/src python -m keep_gpu.mcp.server --mode http --host 'bad host'` raises `socket.gaierror`.

## Solution

- Add focused tests in `tests/mcp/test_server.py` that monkeypatch `server.main()` dependencies and prove invalid HTTP endpoint flags exit before `run_http`.
- Add a small MCP-local endpoint validator in `src/keep_gpu/mcp/server.py`.
- Call that validator only for `--mode http`, immediately before creating `KeepGPUServer` and invoking `run_http`.
- Update `AGENTS.md` to document that both CLI service and MCP executable endpoint flags must be validated before socket bind.

## Tasks

- [x] Add RED tests for invalid MCP executable endpoint flags:
  - `--port 0`
  - `--port 70000`
  - `--port true`
  - `--host "bad host"`
- [x] Add a valid-case test proving normalized `host` and `port` reach `run_http`.
- [x] Run the focused new tests and confirm the invalid host/range tests fail before implementation.
- [x] Implement minimal endpoint validation in `src/keep_gpu/mcp/server.py`.
- [x] Update `AGENTS.md` with the endpoint validation rule.
- [x] Run `pytest tests/mcp/test_server.py -q`.
- [x] Run `git diff --check`.
- [x] Commit as `fix(mcp): validate executable endpoint flags`.

## Verification Notes

Expected final checks:

```bash
pytest tests/mcp/test_server.py -q
git diff --check
```

The RED run should show the new invalid `--host`, `--port 0`, and `--port 70000` tests failing because `run_http` was called instead of argparse exiting. The bool-ish string case is already rejected by argparse's `type=int`, and remains covered as part of the executable contract.
