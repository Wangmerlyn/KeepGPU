# JSON-RPC Request Version Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject explicit JSON-RPC request versions that are not exactly `"2.0"` while preserving legacy direct calls that omit `jsonrpc`.

**Architecture:** The MCP server funnels JSON-RPC and legacy direct method calls through `_handle_request()` in `src/keep_gpu/mcp/server.py`. The fix belongs at the top-level request validation boundary before method dispatch so all MCP lifecycle and tool methods share the same JSON-RPC version contract.

**Tech Stack:** Python, pytest, KeepGPU MCP JSON-RPC server.

---

## Background

The audit found that `_handle_request()` accepts requests such as `{"jsonrpc": "1.0", "id": 1, "method": "tools/list"}` and returns a successful JSON-RPC 2.0 response. Standard JSON-RPC requests that include a `jsonrpc` member must use the exact string `"2.0"`. Existing local scripts and tests also use legacy direct calls without a `jsonrpc` member, so omitted versions must remain supported.

## Solution

Add explicit version validation in `_handle_request()` after confirming the payload is an object and before method dispatch. If `jsonrpc` is present and its value is not `"2.0"`, return `JSONRPC_INVALID_REQUEST` (`-32600`). Do not require `jsonrpc` for legacy/internal direct calls.

## Tasks

- [x] Add RED tests in `tests/mcp/test_server.py` for invalid explicit version, valid explicit `"2.0"`, and omitted legacy version.
- [x] Run `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q` and confirm the invalid-version test fails before implementation.
- [x] Implement the minimal `_handle_request()` validation in `src/keep_gpu/mcp/server.py`.
- [x] Update `AGENTS.md` with the JSON-RPC version compatibility note.
- [x] Run `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py -q`.
- [x] Run `git diff --check`.
- [x] Commit with `fix(mcp): validate jsonrpc request version`.

## Verification Notes

The RED run must fail because explicit `"jsonrpc": "1.0"` currently succeeds. The final targeted pytest run must pass all MCP server tests, and `git diff --check` must report no whitespace errors before commit.
