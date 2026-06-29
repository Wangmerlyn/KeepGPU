# Pending Stop Status Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make timed-out stops of still-starting sessions truthful in `status`, so a remembered cancellation is never exposed as a normal `starting` or `active` keeper.

**Architecture:** Keep the fix inside `KeepGPUServer` lifecycle state. `_pending_stop_job_ids` remains the cancellation memory for starts that have not settled, but status rendering must expose that memory as `state="stopping"` with the existing timeout message, and `start_keep()` must publish a pending-stopped session as `stopping` before launching background cleanup.

**Tech Stack:** Python 3, `threading`, pytest, KeepGPU MCP/service lifecycle tests.

---

### Task 1: Preserve Truthful Pending-Stop State

**Files:**
- Modify: `src/keep_gpu/mcp/server.py`
- Modify: `tests/mcp/test_server.py`
- Modify: `README.md`
- Modify: `docs/concepts/architecture.md`
- Modify: `docs/guides/mcp.md`
- Modify: `docs/reference/cli.md`
- Modify: `AGENTS.md`

- [x] **Step 1: Write failing tests for pending cancellation visibility**

Add or adjust focused tests in `tests/mcp/test_server.py`:

```python
def test_timed_out_stop_of_starting_session_reports_stopping_before_startup_finishes(monkeypatch):
    # Start a controller whose keep() blocks, stop it with a tiny startup wait
    # timeout, and assert status(job_id) plus status() expose the remembered
    # cancellation as state="stopping" with the timeout message while keep()
    # is still blocked.
```

```python
def test_pending_stop_session_is_not_reported_with_active_state_before_background_release(monkeypatch):
    # Defer only the background pending-stop thread started after startup
    # completes, then assert the published session is visible with
    # state="stopping" and the timeout message, not state="active", before
    # running the deferred cleanup thread.
```

- [x] **Step 2: Run tests to verify they fail for the current bug**

Run:

```bash
PYTHONPATH=$PWD/src pytest \
  tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_reports_stopping_before_startup_finishes \
  tests/mcp/test_server.py::test_pending_stop_session_is_not_reported_with_active_state_before_background_release \
  -q
```

Expected: both tests fail because the current server reports the first window as plain `starting` with `last_error=None` and the second window as plain `active`.

- [x] **Step 3: Implement minimal lifecycle-state fix**

In `src/keep_gpu/mcp/server.py`:

- Add a small helper for the timeout status shape or reuse `_timeout_error_message()` directly.
- When `status(job_id)` reports a job from `_starting_params`, check whether the job id is in `_pending_stop_job_ids`; if so, report `state="stopping"` and `last_error=_timeout_error_message()`.
- When all-session `status()` renders `_starting_params`, apply the same rule per job.
- When `start_keep()` settles a startup whose job id is pending stop, create the `Session` with `state="stopping"` and `last_error=_timeout_error_message()` before adding it to `_sessions` and before launching the background cleanup thread.
- Ensure the background pending-stop cleanup path can still release this already-marked `stopping` session, while duplicate user stop requests for ordinary `stopping` sessions still return `timed_out` without starting duplicate release work.
- Keep identity guards intact: the background `_stop_current_session(... expected_session=session)` must still only stop the original session, not a reused job id.

- [x] **Step 4: Verify targeted lifecycle tests**

Run:

```bash
PYTHONPATH=$PWD/src pytest \
  tests/mcp/test_server.py::test_stop_keep_times_out_waiting_for_stuck_starting_session \
  tests/mcp/test_server.py::test_stop_all_times_out_waiting_for_stuck_starting_session \
  tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_reports_stopping_before_startup_finishes \
  tests/mcp/test_server.py::test_pending_stop_session_is_not_reported_with_active_state_before_background_release \
  tests/mcp/test_server.py::test_timed_out_stop_of_starting_session_releases_after_startup_completes \
  tests/mcp/test_server.py::test_pending_stop_does_not_stop_reused_job_id_after_original_removed \
  tests/mcp/test_server.py::test_timed_out_stop_all_of_starting_session_releases_after_startup_completes \
  -q
```

Expected: all listed tests pass.

- [x] **Step 5: Update lifecycle documentation**

Update docs to say that once a stop wait times out for a starting job, `status` shows the remembered cancellation as `state="stopping"` with the timeout message, and after startup succeeds the session remains `stopping` until background release succeeds or becomes `stop_failed`.

- [x] **Step 6: Run broader verification**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/mcp -q
PYTHONPATH=$PWD/src pytest tests -q
pre-commit run --all-files
PYTHONPATH=$PWD/src mkdocs build
git diff --check
```

Expected: tests and checks pass. Existing mkdocs warnings about plan pages outside nav are acceptable if unchanged.

- [x] **Step 7: Commit**

Commit with:

```bash
git add src/keep_gpu/mcp/server.py tests/mcp/test_server.py README.md docs/concepts/architecture.md docs/guides/mcp.md docs/reference/cli.md AGENTS.md docs/plans/pending-stop-status-truth.md
git commit -m "fix(mcp): expose pending startup stops in status"
```
