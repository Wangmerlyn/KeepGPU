# RPC Leading Slash Routes Plan

## Background

Python's `BaseHTTPRequestHandler` collapses request targets that begin with
`//` before KeepGPU route checks read `self.path`. That can make raw targets such
as `//api/sessions` and `//rpc` behave like canonical API/RPC routes.

## Goal

Reject raw leading-double-slash API/RPC aliases as structured JSON
`404 Unknown endpoint` responses before body parsing, JSON-RPC dispatch, or REST
session side effects.

## Tasks

- [x] Add raw-socket regression tests for `POST //api/sessions`,
  `DELETE //api/sessions/{job_id}`, and `POST //rpc`.
- [x] Capture or derive the raw request target before relying on normalized
  `self.path`.
- [x] Reuse the existing noncanonical API/RPC structured 404 helpers.
- [x] Update API/RPC docs and `AGENTS.md` to call out leading `//` aliases.
- [x] Run targeted MCP HTTP tests, full tests, docs build, pre-commit, and
  `git diff --check`.
