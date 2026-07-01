# Encoded RPC Alias Rejection Plan

## Background

KeepGPU's HTTP server reserves exact `/rpc` for JSON-RPC POST requests. Recent
routing hardening rejects trailing slash, parameter, query, and encoded
separator variants so they do not fall through to dashboard/static handling or
JSON-RPC body parsing.

One alias class remains: paths such as `/rp%63` and `/%72pc` decode to exact
`/rpc` but are not the canonical raw endpoint. `GET /rp%63` can reach dashboard
fallback, and unsupported methods can fall through to `BaseHTTPRequestHandler`
HTML responses instead of structured JSON.

## Goal

Reject percent-encoded exact `/rpc` aliases as structured JSON `404 Unknown
endpoint` responses before static fallback, JSON-RPC parsing, or stdlib HTML
errors. Preserve exact raw `/rpc` behavior: POST remains JSON-RPC, and unsupported
methods return structured JSON `405`.

## Solution

- Add RED tests for encoded exact `/rpc` aliases across GET, POST, and
  unsupported methods.
- Treat decoded `/rpc` with raw `parsed.path != "/rpc"` as noncanonical.
- Keep existing `/rpc` and `/` JSON-RPC compatibility behavior unchanged.
- Update agent/docs wording for the encoded alias contract.

## Tasks

- [x] Create an isolated `.worktrees/codex/rpc-encoded-aliases` branch from
  latest `origin/main`.
- [x] Add RED tests for encoded exact `/rpc` aliases.
- [x] Implement the minimal route predicate fix.
- [x] Update `AGENTS.md` and CLI/API docs.
- [x] Run targeted/full verification and local subagent review.
- [ ] Open a PR, resolve hosted comments/checks, squash merge, and clean up.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q -k 'http_rpc_noncanonical_get_returns_json_404_without_static_fallback or http_rpc_encoded_exact_alias'`
  failed with 4 failures: encoded exact GET aliases returned dashboard `200`,
  and encoded exact unsupported methods returned stdlib `501`.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q -k 'http_rpc_noncanonical_get_returns_json_404_without_static_fallback or http_rpc_encoded_exact_alias or http_rpc_get_rejects_with_json_405 or http_jsonrpc_parse_error_returns_jsonrpc_envelope'`
  passed with 13 passed.
- HTTP API:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with 121
  passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 862 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the existing
  Material for MkDocs warning.
- Hooks:
  `pre-commit run --all-files --show-diff-on-failure` and `git diff --check`
  passed after Black reformatted `src/keep_gpu/mcp/server.py`.
- Local review:
  a local subagent review found no must-fix issues and reran the focused
  regression slice with 13 passed.
