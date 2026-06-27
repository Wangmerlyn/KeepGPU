# Stop-All Release Concurrency Plan

## Background

`KeepGPUServer.stop_keep(None)` now keeps lifecycle state truthful, waits for
sessions that were already starting, and preserves deterministic stop-all
snapshot semantics. After the snapshot, however, it releases sessions
sequentially. Each `_release_with_timeout()` call can take up to 10 seconds, so
multiple slow sessions can make `keep-gpu stop --all`, REST, JSON-RPC, or the
dashboard time out before receiving the truthful stop result.

Targeted `stop_keep(job_id)` is intentionally out of scope for this branch.

## Goal

Release independent sessions concurrently during stop-all while preserving the
existing response contract, lifecycle state, late-release callbacks, and stable
result ordering.

## Design

- Keep the current locked stop-all snapshot and `state="stopping"` marking.
- Launch one stop worker per releasable session after the lock is released.
- Each worker calls `_release_with_timeout()` with the same late-result callback
  behavior used today.
- Aggregate worker outcomes in the original stop-all snapshot order, not in
  completion order, so response ordering remains deterministic.
- Continue to report already-`stopping` sessions as `timed_out` without starting
  another release.
- Keep targeted stop behavior unchanged.

## Todo

- [x] Add a failing stop-all test showing release workers enter concurrently.
- [x] Add a failing mixed-result test for concurrent stop-all aggregation:
      success, timeout, and failure must all be reported in snapshot order.
- [x] Add a stop-all late-callback test showing late success removes the session
      and late failure keeps `state="stop_failed"`.
- [x] Make existing stop-all timeout tests independent of release call order.
- [x] Implement concurrent stop-all release workers without holding
      `_sessions_lock` during controller release work.
- [x] Update `AGENTS.md` and docs to state that stop-all releases independent
      sessions concurrently while keeping additive result fields.
- [x] Run targeted MCP tests, full tests, docs build, and pre-commit.
- [ ] Open a GitHub PR, run local subagent review, resolve all comments, then
      squash merge.
