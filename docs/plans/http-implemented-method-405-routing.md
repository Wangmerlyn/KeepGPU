# HTTP Implemented-Method 405 Routing Plan

## Background

The MCP HTTP handler already has a shared unsupported-method helper that returns
structured JSON `405 Method Not Allowed` responses with an `Allow` header for
known routes. That helper is used for BaseHTTPRequestHandler-generated
unsupported verbs and for `GET /rpc`, but `do_POST()` and `do_DELETE()` can still
fall through to local `404 Unknown endpoint` branches when the path is known but
the implemented verb is not allowed.

## Goal

Known HTTP routes must return the same structured JSON `405` response whenever a
client uses a wrong method, including wrong methods that are implemented by the
server class. Unknown `/api/*` routes must keep returning structured JSON `404`
responses.

## Solution

- Add RED REST/RPC tests for `POST` and `DELETE` against known routes that do
  not allow those methods.
- Route known-but-wrong methods in `do_POST()` and `do_DELETE()` through the
  existing `_send_api_rpc_unsupported_method_response()` helper before reading
  request bodies or returning local unknown-endpoint `404`s.
- Update `AGENTS.md` with the implemented-handler maintenance rule.

## Tasks

- [x] Create an isolated `.worktrees/codex/http-unsupported-method-routing`
  branch from latest `origin/main`.
- [x] Add RED tests for implemented handlers returning `404` on known wrong
  methods.
- [x] Implement the minimal handler routing fix.
- [x] Run targeted MCP HTTP tests.
- [x] Run broader verification, docs build, and pre-commit.
- [x] Request local subagent review and resolve findings.
- [ ] Open a PR, resolve hosted review comments, wait for green checks, squash
  merge, and clean up the worktree/branches.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_implemented_handlers_reject_known_wrong_methods_with_json_405 -q`
  failed with all seven parametrized cases returning `404` instead of `405`.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_implemented_handlers_reject_known_wrong_methods_with_json_405 -q`
  passed with 7 passed.
- MCP:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with 177 passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 590 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan-page notices.
- Local review:
  a local subagent review found no Critical, Important, or Minor issues and
  recommended marking this checklist item complete before PR publication.
