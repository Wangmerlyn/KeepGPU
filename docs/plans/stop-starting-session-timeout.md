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
  completes successfully, immediately trigger a quiet background stop so a
  timed-out cancellation does not leave a surprise active keeper.
- Preserve existing behavior for slow startups that settle within the wait
  budget: the original stop request releases the session and reports it in
  `stopped`.
- Keep failed startups clearing reservations and pending-stop markers.
- Update `AGENTS.md`, MCP/CLI docs, and this plan.

## Tasks

- [x] Add RED tests for targeted stop and stop-all timing out while startup is
      stuck without hanging the test process.
- [x] Add GREEN coverage that a timed-out stop request is honored after startup
      eventually completes.
- [x] Add GREEN coverage that stop-all does not duplicate a starting timeout if
      startup settles between timeout classification and the session snapshot.
- [x] Add GREEN coverage that a timed-out stop followed by startup failure clears
      pending-stop state before a later reuse of the same job ID.
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
  first failed because stop threads remained blocked on the unbounded startup
  wait. After implementation, this shard passed with `3 passed`, and the
  existing slow-start preservation shard
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_keep_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_only_for_sessions_starting_at_snapshot -q`
  passed with `3 passed`.
- Race regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_all_does_not_duplicate_starting_timeout_after_startup_settles -q`
  first failed because `timed_out` contained `starting-job` twice. After the
  release-loop guard, it passed with `1 passed`.
- Failed-start pending-stop regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_timed_out_stop_of_failed_startup_does_not_affect_reused_job_id -q`
  failed under a temporary mutation that removed pending-stop cleanup from the
  startup failure path, then passed with `1 passed` after restoring cleanup.
- Combined focused shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py::test_stop_keep_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_stop_all_times_out_waiting_for_stuck_starting_session tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_releases_after_startup_completes tests/mcp/test_server.py::test_timed_out_stop_of_failed_startup_does_not_affect_reused_job_id tests/mcp/test_server.py::test_stop_all_does_not_duplicate_starting_timeout_after_startup_settles tests/mcp/test_server.py::test_stop_keep_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_for_starting_session tests/mcp/test_server.py::test_stop_all_waits_only_for_sessions_starting_at_snapshot -q`
  passed with `8 passed`.
- MCP shard:
  `PYTHONPATH=$PWD/src pytest tests/mcp -q` passed with `184 passed`.
- Full no-GPU-safe gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `626 passed, 11 skipped`.
- Docs and hygiene:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan notices; `pre-commit run --all-files` passed; and
  `git diff --check` passed. Installing the full stale
  `requirements_dev.txt` failed on Python 3.12 because `watchdog==0.9.0`
  imports the removed `imp` module, so the missing gate tools were installed
  directly as `pre-commit mkdocs mkdocs-material mkdocstrings[python]`.
- Local subagent review:
  Reviewer found stale AGENTS wording, stale verification counts, and the
  missing failed-start pending-stop coverage. These were addressed before PR.
  Reviewer re-check found no new must-fix issues; the only remaining note was
  to ensure this plan file is staged for the PR.
