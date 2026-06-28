# CLI Unconditional Threshold Log Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make blocking-mode CLI logs describe `busy_threshold=-1` as unconditional mode instead of rendering it as `-1%`.

**Architecture:** Keep validation and controller behavior unchanged. Add only a narrow log-formatting path in `src/keep_gpu/cli.py`, with focused CLI coverage in `tests/test_cli_thresholds.py`.

**Tech Stack:** Python, Typer CLI, pytest, standard logging capture by way of `caplog`.

---

## Background

Blocking mode currently validates `busy_threshold` and then logs it with
`logger.info("Busy threshold: %s%%", busy_threshold)`. That is correct for normal
percentage thresholds such as `25`, but misleading for the sentinel value `-1`,
which means explicit unconditional keepalive mode with utilization backoff
disabled.

The existing product guidance already treats sentinel values as semantic labels
for dashboard UI. This change applies the same clarity to blocking-mode CLI logs
without changing CLI validation or controller inputs.

## Solution

- Add a focused failing test around `_run_blocking(..., busy_threshold=-1)` that
  proves the log contains semantic unconditional wording and does not contain
  `Busy threshold: -1%`.
- Cover the normal threshold path so `busy_threshold=25` still logs `25%`.
- Update the CLI logging line to format only `-1` semantically and keep
  non-negative thresholds as percentages.
- Add a CLI-focused sentinel-log note to `AGENTS.md` if the existing dashboard
  note is too narrow.

## Tasks

- [x] Add the RED test in `tests/test_cli_thresholds.py`.
- [x] Run the focused test and record the expected failure.
- [x] Implement the minimal CLI logging formatter/change in `src/keep_gpu/cli.py`.
- [x] Update `AGENTS.md` with blocking-mode log guidance.
- [x] Run:
  - `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py -q`
  - `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  - `git diff --check`
  - `pre-commit run --all-files` if available and cheap
- [x] Strengthen the hosted-review follow-up assertion so forbidden messages
      are checked as substrings of captured log entries.
- [x] Commit with `fix(cli): label unconditional threshold logs`.

## Verification Notes

- RED focused test:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py::test_run_blocking_logs_busy_threshold_semantically -q`
  failed before the production change because captured logs contained
  `Busy threshold: -1%` instead of the semantic unconditional message.
- GREEN focused test:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py::test_run_blocking_logs_busy_threshold_semantically -q`
  passed with 2 tests after the CLI log change.
- Full threshold suite:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py -q` passed with
  14 tests.
- Relevant CLI suite:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with 136 tests.
- Hygiene:
  `git diff --check` passed, and `pre-commit run --all-files` passed.
- Hosted review follow-up:
  Gemini noted the forbidden-message check needed substring matching. The test
  now checks `expected_message` and `forbidden_message` against captured log
  entry substrings before re-running verification.
- The final diff avoids controller behavior changes and keeps all changes scoped
  to CLI logging, tests, docs guidance, and this plan.
