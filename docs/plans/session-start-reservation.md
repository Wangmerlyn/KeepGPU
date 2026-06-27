# Session Start Reservation Implementation Plan

## Background

`KeepGPUServer.start_keep()` checks whether a `job_id` is active, releases the
session lock, starts controller work, then checks the map again. Two concurrent
requests with the same custom `job_id` can both start controller work; one only
loses after keep-alive work has already begun. The losing cleanup path also
risks doing controller release work while holding the session lock.

Stop-all release parallelism is a separate issue and is intentionally out of
scope for this branch.

## Goal

Reject duplicate custom `job_id` requests while the first request is still
starting, before the duplicate constructs or starts any controller work.

## Solution

- Add a `_starting_job_ids` reservation set guarded by `_sessions_lock`.
- Reserve the final `job_id` before constructing or starting a controller.
- Reject duplicates when a `job_id` is either active or reserved.
- Remove the reservation in all success and failure paths.
- Use a condition on the session lock so targeted and stop-all requests wait
  for starting sessions to settle before deciding what can be released.
- Keep controller `keep()` and `release()` work outside `_sessions_lock`.

## Todo

- [x] Add failing server tests showing a concurrent duplicate `job_id` is
      rejected before the duplicate controller starts.
- [x] Add a cleanup test showing a failed start removes the reservation so the
      same `job_id` can be retried.
- [x] Add stop tests showing targeted and stop-all requests do not miss a
      session that is still starting.
- [x] Add stop-all snapshot coverage so starts created after stop-all begins
      are not waited on or stopped by that request.
- [x] Implement `_starting_job_ids` reservation in `KeepGPUServer.start_keep()`.
- [x] Update `AGENTS.md` and service docs for custom `job_id` uniqueness during
      active and starting sessions.
- [x] Run targeted MCP tests, full tests, docs build, and pre-commit.
- [ ] Open a GitHub PR, run local subagent review, resolve all comments, then
      squash merge.
