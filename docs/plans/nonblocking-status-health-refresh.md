# Nonblocking Status Health Refresh Plan

## Background

`KeepGPUServer.status()` refreshes runtime failure state by calling controller
health hooks. Those hooks are intended to be lightweight, but a slow or wedged
hook currently runs while the session lock is held, so status can block stop,
start, and stop-all lifecycle operations.

## Goal

Keep status observational: runtime-health probing must not hold the service
session lock, and a late probe result must not mutate a session that has already
been stopped or replaced.

## Solution

- Add a regression test with a blocking `runtime_error()` hook and prove
  `stop_keep(job_id)` still completes while status is waiting on the hook.
- Snapshot the target session under the lock, run the runtime-health hook outside
  the lock, then reacquire the lock before applying a runtime-failed state.
- Render status from the current session state after the probe so stop and
  replacement races stay truthful.
- Update agent and MCP/service docs to preserve the lifecycle-lock contract.

## Verification

- Targeted MCP server lifecycle tests.
- Focused MCP service tests.
- Pre-commit, docs build, and diff whitespace checks before PR.
