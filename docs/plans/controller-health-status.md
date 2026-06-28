# Controller Health Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep service status truthful when an already-started GPU keep worker later reaches a terminal allocation failure.

**Architecture:** Add a tiny read-only runtime-health contract from single-GPU controllers to `GlobalGPUController`, then have MCP/REST status refresh active sessions from that contract. The check must report only terminal worker failures, not eco-safe allocation deferral caused by busy or unavailable telemetry.

**Tech Stack:** Python, pytest, KeepGPU controllers, MCP/REST service docs.

---

## Background

CUDA and ROCm workers signal startup before first tensor allocation so the
service can remain responsive and avoid forcing immediate work on busy GPUs. That
eco-safe behavior is correct, but status can remain `active` even after a worker
has stopped permanently because allocation retries were exhausted or an internal
terminal failure was captured.

## Solution

- Add no-GPU tests for a read-only `GlobalGPUController.runtime_error()` helper
  that returns the first terminal child controller error and ignores controllers
  that do not implement the optional hook.
- Add MCP status tests showing a session moves from `active` to
  `runtime_failed` with `last_error`, without calling `release()` or deleting the
  session.
- Add dashboard helper coverage so the browser UI displays a humane
  `runtime_failed` label and retained error detail.
- Keep startup failures synchronous and unchanged.
- Document that `runtime_failed` is a retained, inspectable session state and
  that deferred allocation is not a failure.

## Todo

- [x] Add failing global-controller runtime-health tests.
- [x] Add failing MCP status tests for single-session and list status.
- [x] Add failing dashboard helper test for retained runtime failures.
- [x] Verify the new tests fail on current behavior.
- [x] Implement the smallest runtime-health contract.
- [x] Update the dashboard helper and packaged static asset.
- [x] Update `AGENTS.md`, MCP guide, CLI reference, and architecture docs.
- [x] Run targeted tests, broader tests, docs build, pre-commit, and diff checks.
- [x] Run local subagent code review and resolve all must-fix findings.
- [x] Resolve external PR review comments for stable runtime failure retention
  and defensive health-hook exceptions.
- [ ] Open PR, resolve GitHub review/check feedback, squash merge, and clean up.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/global_controller/global_keep_test.py tests/mcp/test_server.py -q`
- `PYTHONPATH=$PWD/src pytest tests/mcp tests/global_controller tests/single_gpu_controller -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `npm --prefix web/dashboard test -- src/lib/session.test.js`
- `npm --prefix web/dashboard test`
- `npm --prefix web/dashboard run build`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
- `git diff --check`
