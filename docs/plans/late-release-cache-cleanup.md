# Late Release Cache Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for the regression and superpowers:verification-before-completion before claiming completion. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CUDA, ROCm, and MPS release paths clean backend caches when a timed-out worker exits before a later release attempt.

**Architecture:** Keep the fix local to each single-GPU controller release method. Preserve the current live-thread path: set the stop event, join once, clean cache only when the thread stops, and raise `TimeoutError` when it remains alive. Add one idempotent late-cleanup branch for stale `_thread` plus set `_stop_evt`.

**Tech Stack:** Python, pytest, PyTorch cache APIs, KeepGPU single-GPU controllers.

---

## Background

`CudaGPUController.release()`, `RocmGPUController.release()`, and `MacMGPUController.release()` currently return early when no worker thread is alive. If the first release attempt times out, it sets `_stop_evt` and raises before backend cache cleanup. When the same thread exits shortly after that timeout, a second release sees a dead thread and returns as "not running", leaving backend cache cleanup skipped and stale `_thread` / `_stop_evt` references in place.

ROCm already guarantees `_shutdown_rocm_smi()` through a `finally` block in release paths, and that behavior must remain unchanged.

## Solution

1. Add regression tests in `tests/single_gpu_controller/test_release_contract.py` that simulate a worker surviving the first join, then dying before a second release call.
2. Verify those tests fail before controller changes because cleanup is skipped and stale state remains.
3. Add a minimal late-cleanup branch to each controller release method:
   - only when `_thread` exists,
   - the thread is no longer alive,
   - `_stop_evt` exists and is already set.
4. In that branch, run backend cache cleanup, clear `_thread` and `_stop_evt`, log an info message, and return.
5. Preserve warning behavior for no thread / not stopping states.
6. Keep ROCm SMI shutdown in all release paths.

## Task Checklist

- [x] Inspect existing release methods and release-contract tests.
- [x] Add failing late-cleanup regression tests before production code changes.
- [x] Record RED evidence in this plan.
- [x] Implement CUDA late cleanup.
- [x] Implement ROCm late cleanup while preserving `_shutdown_rocm_smi()`.
- [x] Implement MPS late cleanup.
- [x] Update `AGENTS.md` with the release idempotency lifecycle invariant.
- [x] Add post-review coverage for dead-but-not-stopping workers that must keep
  the existing not-running warning behavior.
- [x] Address hosted review feedback so successful normal release also clears
  stale thread and stop-event state.
- [x] Run focused GREEN verification and record evidence.
- [x] Run broader targeted tests, docs build, pre-commit, and diff check.
- [x] Commit with `fix(controllers): clean caches after delayed release`.

## RED Evidence

Command:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py -q
```

Result: failed as expected before production changes.

Key failures:

- CUDA late release: `cache_calls == []`, expected `["empty_cache"]`; second release logged `keep thread not running`.
- ROCm late release: `cache_calls == []`, expected `["empty_cache"]`; second release logged `keep thread not running`.
- MPS late release: `cache_calls == []`, expected `["empty_cache", "gc.collect"]`; second release logged `keep thread not running`.

Summary: `3 failed, 5 passed`.

Hosted review follow-up:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py::test_release_success_clears_state_so_second_release_keeps_not_running_behavior -q
```

Result: failed before the follow-up fix with `3 failed`. A normal successful
release cleaned the backend cache but left `_thread` and `_stop_evt` populated,
so a later release could be mistaken for late cleanup.

## GREEN Evidence

Focused command:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py -q
```

Result: passed after controller changes. After local review, the focused command
was expanded with negative coverage for dead threads whose stop event is missing
or not set; those paths preserve the existing not-running behavior and skip
cache cleanup.

Summary: `17 passed`.

Hosted review follow-up:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py::test_release_success_clears_state_so_second_release_keeps_not_running_behavior -q
```

Result: `3 passed`.

## Final Verification Evidence

Focused regression:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py -q
```

Result: `17 passed`.

Broader targeted:

```bash
PYTHONPATH=$PWD/src pytest tests/single_gpu_controller/test_release_contract.py tests/cuda_controller/test_keep_and_release.py tests/rocm_controller/test_rocm_backoff.py tests/macm_controller -q
```

Result: `39 passed, 6 skipped`.

Documentation build:

```bash
PYTHONPATH=$PWD/src mkdocs build
```

Result: succeeded; MkDocs reported existing nav warnings for plan files not included in navigation.

Pre-commit:

```bash
pre-commit run --all-files
```

Result: all hooks passed.

Whitespace check:

```bash
git diff --check
```

Result: passed with no output.
