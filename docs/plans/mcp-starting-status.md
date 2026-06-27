# MCP Starting Status Plan

## Background

`KeepGPUServer.start_keep()` reserves a custom `job_id` before starting the
controller, but `status()` only reads `_sessions`. While `controller.keep()` is
still running, `status(job_id)` reports inactive and `status()` omits the job,
even though startup work has begun and duplicate starts are already rejected.

Stop paths already wait for `_starting_job_ids`, so the inconsistency is in
status visibility rather than stop safety.

## Goal

Make starting sessions visible as `state="starting"` with their requested
params, so humans, CLIs, dashboards, and agents do not mistake in-progress
startup for no session.

## Design

- Keep the existing `_starting_job_ids` set for stop/wait behavior.
- Add a guarded starting-params map keyed by `job_id`.
- Insert starting params under `_sessions_lock` at the same moment the job ID is
  reserved.
- Remove starting params on startup failure or when the session is promoted to
  active.
- Make `status(job_id)` return `active=True`, `state="starting"`, and
  `last_error=None` while startup is in progress.
- Make `status()` include starting jobs in `active_jobs` with the same shape as
  active/stopping/failed sessions.
- Update docs and `AGENTS.md` with the truthfulness rule.

## Todo

- [x] Add a failing MCP server test for `status(job_id)` and `status()` during
      blocked controller startup.
- [x] Implement starting params tracking in `src/keep_gpu/mcp/server.py`.
- [x] Update MCP docs, CLI reference, README, and `AGENTS.md`.
- [x] Run targeted MCP/service tests, full tests, docs build, pre-commit, and
      local subagent review.
- [ ] Open PR, resolve remote review/checks, squash merge, and clean up the
      worktree branch.
