# API Encoded Path JSON Errors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep encoded API-shaped HTTP routes inside the structured JSON API error boundary.

**Architecture:** Reuse the existing noncanonical-route pattern from `/rpc`. Treat raw paths that are not canonical API routes but whose raw or decoded target is API-shaped (`/api`, `/api/...`, `/api;...`, `/api?...`, or `/api#...`) as noncanonical API routes, returning REST-shaped JSON `404 Unknown endpoint` before dashboard/static fallback or `BaseHTTPRequestHandler` HTML errors.

**Tech Stack:** Python, `http.server`, pytest, MkDocs.

---

## Tasks

- [x] Add RED HTTP tests proving encoded API route spellings such as `GET /api%2Fsessions`, `GET /api%3Bdebug`, `GET /api%3Fsessions`, and `OPTIONS /api%3Bdebug` return HTML or wrong statuses on current `main`.
- [x] Add decoded noncanonical API-route detection to `_JSONRPCHandler`.
- [x] Route GET, POST, DELETE, and unsupported-method handling through the new guard while preserving canonical encoded `job_id` validation.
- [x] Update `AGENTS.md`, HTTP reference docs, and this plan with the routing contract.
- [x] Run targeted suite, full verification, docs build, diff check, and pre-commit before PR.
- [x] Run local subagent review before PR.

## Verification

- Baseline: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with `95 passed`.
- Red: expanded encoded API route shard failed with three `200` dashboard responses and two `501` stdlib HTML unsupported-method responses.
- Green: expanded encoded API route shard passed with `17 passed`; nearby routing compatibility shard passed with `12 passed`.
- HTTP API suite: `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py -q` passed with `113 passed`.
- Full suite: `PYTHONPATH=$PWD/src pytest -q` passed with `821 passed, 11 skipped`.
- Docs/build: `PYTHONPATH=$PWD/src mkdocs build --strict` passed; Material for MkDocs emitted its upstream MkDocs 2.0 warning.
- Hygiene: `git diff --check` passed; `pre-commit run --all-files --show-diff-on-failure` passed.
