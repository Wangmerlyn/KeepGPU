# Service Live PID Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent service auto-start from overwriting the PID record of an already-running KeepGPU daemon whose health endpoint is unavailable.

**Architecture:** Keep daemon ownership decisions centralized in `src/keep_gpu/cli.py`. `_ensure_service_running()` should distinguish dead stale PID records from live ownership-verified records before spawning another daemon; only dead records are cleared for auto-start.

**Tech Stack:** Python, Typer CLI helpers, pytest, MkDocs.

---

## Task 1: Refuse Auto-Start Over a Live Managed Daemon

**Files:**
- Modify: `src/keep_gpu/cli.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `AGENTS.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/reference/cli.md`
- Create: `docs/plans/service-live-pid-guard.md`

- [x] **Step 1: Reproduce the overwrite risk with a failing regression**

Add a focused test where `_service_available()` is false, the PID file describes
an ownership-verified live KeepGPU daemon, and auto-start would otherwise spawn
over the record.

Run:

```bash
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'live_managed_pid'
```

Expected before the fix: the test fails because `_ensure_service_running()`
calls `_start_service_process()` instead of refusing to overwrite the live
record.

- [x] **Step 2: Guard live managed daemon records before auto-start**

In `_ensure_service_running()`, read the full PID record after health fails. If
the recorded PID is dead, clear the stale record and continue. If the recorded
PID is alive and `_record_matches_running_process()` verifies it as the KeepGPU
daemon for the requested host/port, raise a clear `RuntimeError` instructing the
user to inspect the exact service log path or use `keep-gpu service-stop
--force`.

- [x] **Step 3: Update docs and agent contract**

Document that auto-start does not overwrite a live managed daemon record when
health is unavailable.

- [x] **Step 4: Verify**

Run:

```bash
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'live_managed_pid'
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'clears_dead_pid_record'
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'service_process or service_stop or pid_record or ownership or ensure_service_running'
PYTHONPATH=src pytest tests/test_cli_service_commands.py -q
PYTHONPATH=src pytest tests -q
PYTHONPATH=src mkdocs build --strict
pre-commit run --all-files --show-diff-on-failure
git diff --check
```
