# Stop Starting Session Timeout Plan

## Background

`stop_keep(job_id)` and `stop_keep()` intentionally wait for in-progress
`start_keep()` calls so stop requests do not miss sessions that are still
starting. The wait is currently unbounded. If controller construction or
`keep()` hangs, stop requests can block indefinitely, leaving users unable to
cancel a stuck startup and keeping custom job IDs reserved.

## Goal

Keep the no-missed-starts contract for normal slow startups while bounding stop
requests when startup does not settle. If a stop request times out while a job
is still starting, the eventual completed startup should still be released
automatically.

## Solution

- Add bounded waiting for starting jobs in targeted stop and stop-all paths.
- Return the existing additive timeout payload (`timed_out`, `message`) when a
  requested starting job does not settle before the startup-stop wait budget.
- Track timed-out stop requests for starting jobs. If such a startup later
  completes successfully, immediately trigger a logged background stop so a
  timed-out cancellation does not leave a surprise active keeper.
- Preserve existing behavior for slow startups that settle within the wait
  budget: the original stop request releases the session and reports it in
  `stopped`.
- Scope stop-all to the active/starting boundary captured when the stop-all
  request begins. Use session/parameter identity so a later session that reuses
  the same `job_id` is not released by the older stop-all request.
- Bind remembered post-timeout cleanup to the `Session` object created by the
  timed-out startup so a delayed cleanup cannot stop a later replacement
  using the same custom `job_id`.
- Bind targeted stop release to the session observed before the startup wait, or
  to the settled session whose params match the captured starting job, so a stop
  request cannot release a later same-`job_id` replacement.
- Keep failed startups clearing reservations and pending-stop markers.
- Update `AGENTS.md`, MCP/CLI docs, and this plan.

## Tasks

- [x] Add RED tests for targeted stop and stop-all timing out while startup is
      stuck without hanging the test process.
- [x] Add GREEN coverage that a timed-out stop request is honored after startup
      eventually completes.
- [x] Add regression coverage that stop-all does not stop sessions started after
      its initial boundary, including same-`job_id` reuse.
- [x] Add regression coverage that delayed remembered cleanup does not stop a
      later same-`job_id` replacement after the original session was removed.
- [x] Add CodeRabbit follow-up regression coverage that targeted stop does not
      stop a later same-`job_id` replacement after the wait window.
- [x] Implement bounded startup wait and pending-stop handoff.
- [x] Update `AGENTS.md`, MCP/CLI docs, and this plan.
- [x] Run focused tests, MCP shard, full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent code review before PR.
- [ ] Open PR, resolve all review comments/checks, squash merge, and clean the
      worktree.

## Verification

- RED and GREEN focused tests:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_keep_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_stop_all_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_releases_after_startup_completes -q`
  first failed with all stop threads still alive after the short test wait
  budget. After implementation, the expanded focused shard
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_keep_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_stop_all_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_releases_after_startup_completes tests/mcp/test_server.py::test_timed_out_stop_all_of_starting_session_releases_after_startup_completes tests/mcp/test_server.py::test_stop_keep_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_only_for_sessions_starting_at_snapshot -q`
  passed with `7 passed`.
- MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with `183 passed`.
- Full no-GPU-safe gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `625 passed, 11 skipped`.
- Docs and hygiene:
  Initial `pre-commit run --all-files` reformatted `src/keep_gpu/mcp/server.py`
  and `tests/mcp/test_server.py`; after rerunning the focused, MCP, and full
  test shards, the post-review focused shard including the stop-all snapshot
  race regression passed with `8 passed`, `PYTHONPATH=$PWD/src pytest tests/mcp -q`
  passed with `184 passed`, and `PYTHONPATH=$PWD/src pytest tests -q` passed
  with `626 passed, 11 skipped`.
- Same-`job_id` reuse regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_all_does_not_stop_reused_job_id_started_after_snapshot -q`
  failed under the older job-id-only active snapshot filter with
  `stopped == ['reused-job', 'starting-job']`. Restoring the identity-based
  active snapshot made the same command pass with `1 passed`.
- Delayed cleanup same-`job_id` reuse regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_pending_stop_does_not_stop_reused_job_id_after_original_removed -q`
  failed before the identity-bound cleanup helper because the replacement
  controller was released. After binding cleanup to the original `Session`, the
  same command passed with `1 passed`.
- Targeted stop same-`job_id` reuse regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_keep_does_not_stop_reused_job_id_after_wait_window -q`
  failed before the CodeRabbit follow-up because the replacement controller was
  released and the result had no `job_id not found` message. After passing the
  captured expected session into `_stop_current_session()`, the same command
  passed with `1 passed`.
- CodeRabbit follow-up focused shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_all_does_not_stop_session_started_after_snapshot_even_if_it_completes tests/mcp/test_server.py::test_stop_all_does_not_stop_reused_job_id_started_after_snapshot tests/mcp/test_server.py::test_stop_keep_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_stop_all_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_releases_after_startup_completes tests/mcp/test_server.py::test_pending_stop_does_not_stop_reused_job_id_after_original_removed tests/mcp/test_server.py::test_stop_keep_does_not_stop_reused_job_id_after_wait_window tests/mcp/test_server.py::test_timed_out_stop_all_of_starting_session_releases_after_startup_completes tests/mcp/test_server.py::test_stop_keep_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_only_for_sessions_starting_at_snapshot -q`
  passed with `11 passed`.
- Current post-review gates:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with `187 passed`;
  `PYTHONPATH=$PWD/src pytest tests -q` passed with
  `629 passed, 11 skipped`; `PYTHONPATH=$PWD/src mkdocs build`,
  `pre-commit run --all-files`, and `git diff --check` passed.
- Local subagent review:
  The reviewer found a stop-all snapshot race where a session that started
  after stop-all began but completed before the original starting job settled
  could be stopped by that request, plus stale README/architecture docs. Added
  a regression for the race, limited stop-all release work to the initial
  snapshot, and updated the stale docs.
  A later review found a same-`job_id` reuse variant and stale
  AGENTS/CLI wording; the current changes add that regression, keep active
  snapshot membership identity-based, and clarify that stop-all records its
  boundary before waiting.
  Final local reviewers found one more same-`job_id` race on delayed
  remembered cleanup; the current changes add the deterministic regression and
  route pending cleanup through an expected-session helper.
  The final local reviewer reported no Critical or Important findings after
  that fix.
  CodeRabbit later found the targeted-stop variant of the same identity race;
  the current changes add the regression, use the existing expected-session
  helper for targeted stops, and document the guard in `AGENTS.md`.
