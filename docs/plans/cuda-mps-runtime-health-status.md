# CUDA/MPS Runtime Health Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface fatal post-start CUDA and MPS worker failures through `allocation_status()` so service status can retain `state="runtime_failed"`.

**Architecture:** Match the existing ROCm health-hook shape with a controller-local `_failure_exc` field and a non-blocking `allocation_status()` reader. CUDA and MPS should continue treating busy or unavailable telemetry and out-of-memory allocation `RuntimeError` retries as normal backoff; non-OOM allocation or steady-state `RuntimeError` failures, unexpected fatal worker exceptions, and invalid post-start initialized state become retained runtime failures.

**Tech Stack:** Python, pytest, KeepGPU single-GPU controllers, MCP service status contract.

---

## Background

`KeepGPUServer.status()` already checks `GlobalGPUController.runtime_error()`, and the global controller already asks child controllers for `allocation_status()`. ROCm implements that hook and retains terminal allocation failures. CUDA and MPS currently lack the hook and only log unexpected post-start worker exceptions, which can leave service sessions looking active after the worker has failed.

## Solution

- Add no-GPU regression tests for CUDA and MPS direct worker loops.
- Add `_failure_exc: Optional[Exception] = None` to CUDA and MPS controllers.
- Reset `_failure_exc` at the start of `keep()` before a new worker launch.
- Add `allocation_status()` to CUDA and MPS controllers, matching ROCm's read-only health hook.
- Record retained `RuntimeError` details for non-OOM allocation and steady-state `RuntimeError` failures, unexpected non-`RuntimeError` worker exceptions, and invalid direct-loop initialized state when the synchronous startup-error path is not active.
- Leave out-of-memory allocation `RuntimeError` retry behavior unchanged for CUDA/MPS.

## Tasks

- [x] Add failing CUDA regression test for unexpected post-start worker exception retention.
- [x] Add failing MPS regression test for unexpected post-start worker exception retention.
- [x] Run targeted RED tests and record evidence below.
- [x] Implement minimal CUDA health hook and failure capture.
- [x] Implement minimal MPS health hook and failure capture.
- [x] Add the durable `AGENTS.md` invariant.
- [x] Run targeted and full GREEN verification.
- [x] Commit with `fix(controllers): surface cuda mps runtime failures`.

## Local Review Follow-up

Reviewer findings addressed on branch `codex/cuda-mps-runtime-health-status`:

- CUDA and MPS steady-state compute `RuntimeError` failures after successful allocation were still hidden from `allocation_status()`. Non-OOM `RuntimeError` now becomes a retained runtime failure, while out-of-memory `RuntimeError` keeps the existing cache-clear/sleep/retry behavior.
- CUDA and MPS `release()` skipped backend cleanup when a worker was already dead with `_failure_exc` set but `_stop_evt` was missing or unset. Release now performs backend cleanup and clears `_thread`/`_stop_evt` for dead runtime-failed workers, while preserving the existing dead-thread-without-failure "not running" behavior.
- Added a small global status-path assertion that a child health error already prefixed with `rank N:` is returned as-is by `GlobalGPUController.runtime_error()`.

Follow-up tasks:

- [x] Add failing CUDA regression test for post-allocation non-OOM `RuntimeError` retention.
- [x] Add failing MPS regression test for post-allocation non-OOM `RuntimeError` retention.
- [x] Add failing CUDA/MPS release-contract regression test for dead runtime-failed workers with missing or unset stop events.
- [x] Add global runtime-health passthrough assertion for rank-prefixed child failures.
- [x] Run requested RED test command before production edits.
- [x] Implement minimal CUDA/MPS fixes.
- [x] Run focused and broader GREEN verification.
- [x] Address local code-quality review P2 for allocation-time non-OOM
      `RuntimeError` failures.
- [x] Update durable docs to distinguish out-of-memory allocation retries from
      fatal non-OOM allocation failures.

## Verification Log

- RED targeted tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_unexpected_post_start_worker_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_unexpected_post_start_worker_failure -q`
  failed with 2 failures. CUDA and MPS both logged the unexpected
  `ValueError("fatal compute exploded")`, then called the fake stop-event
  `wait()`, which raised `AssertionError("fatal worker failures should stop immediately")`.
- RED gap tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_unexpected_post_start_allocation_failure tests/cuda_controller/test_throttle.py::test_cuda_records_invalid_post_start_num_elements_without_startup_errors tests/macm_controller/test_macm_backoff.py::test_macm_records_unexpected_post_start_allocation_failure -q`
  failed with 3 failures. CUDA/MPS allocation `ValueError("allocator corrupted")`
  escaped from `torch.rand`, and CUDA invalid direct-loop `num_elements`
  returned `allocation_status() is None`.
- GREEN focused regression tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_unexpected_post_start_worker_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_unexpected_post_start_worker_failure tests/cuda_controller/test_throttle.py::test_cuda_records_unexpected_post_start_allocation_failure tests/cuda_controller/test_throttle.py::test_cuda_records_invalid_post_start_num_elements_without_startup_errors tests/macm_controller/test_macm_backoff.py::test_macm_records_unexpected_post_start_allocation_failure -q`
  passed with `5 passed`.
- GREEN targeted controller/service tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/macm_controller tests/rocm_controller/test_rocm_backoff.py tests/mcp/test_server.py -q`
  passed with `135 passed, 9 skipped`.
- Full tests: `PYTHONPATH=$PWD/src pytest tests -q` passed with
  `545 passed, 11 skipped`.
- Docs build: `PYTHONPATH=$PWD/src mkdocs build` passed. It emitted the repository's
  existing Material for MkDocs notice and listed plan pages not included in nav.
- Pre-commit: `pre-commit run --all-files` passed after the first pass
  reformatted the CUDA and MPS test files.
- Diff check: `git diff --check` passed.
- Local review RED tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_cuda_mps_worker -q`
  failed with `6 failed`. CUDA and MPS non-OOM steady-state `RuntimeError`
  failures called the forbidden wait path instead of returning immediately with
  `_failure_exc`; CUDA/MPS dead runtime-failed release cases warned
  `keep thread not running` and did not call backend cleanup.
- Local review focused GREEN tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_cuda_mps_worker -q`
  passed with `6 passed in 1.22s`.
- Local review broader controller/status tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/macm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  initially failed with 4 existing dead-thread-without-failure contract cases
  because release read `_failure_exc` directly on `__new__`-constructed
  controllers. After narrowing that check to `getattr(..., None)`, the same
  command passed with `52 passed, 10 skipped in 1.29s`.
- Coordinator final focused tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_cuda_mps_worker -q`
  passed with `6 passed in 1.15s`.
- Coordinator final broader controller/status tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/macm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  passed with `52 passed, 10 skipped in 1.26s`.
- Coordinator final full tests:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `552 passed, 11 skipped`.
- Coordinator final docs build:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material for MkDocs
  notice and the repository's existing list of plan pages not included in nav.
- Coordinator final pre-commit:
  `pre-commit run --all-files` passed.
- Coordinator final diff check:
  `git diff --check` passed.
- Local code-quality review P2 RED tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_allocation_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_allocation_runtime_error_as_failure -q`
  failed with 2 failures. CUDA reached the forbidden wait path for
  `RuntimeError("illegal memory access")`, and MPS attempted non-OOM cache
  cleanup for `RuntimeError("mps allocation exploded")` instead of retaining a
  runtime failure.
- Local code-quality review P2 GREEN tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_allocation_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_allocation_runtime_error_as_failure -q`
  passed with `2 passed in 1.17s`.
- Final combined focused runtime-health tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_runtime_error_as_failure tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_allocation_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_allocation_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_cuda_mps_worker tests/global_controller/global_keep_test.py::test_global_runtime_error_preserves_rank_prefixed_child_failure -q`
  passed with `9 passed in 1.18s`.
- Final broader controller/status tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/macm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  passed with `54 passed, 10 skipped in 1.26s`.
- Final full tests after local review P2 fix:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `554 passed, 11 skipped`.
- OOM retry coverage tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_retries_post_start_allocation_oom_without_failure tests/cuda_controller/test_throttle.py::test_cuda_retries_steady_state_oom_without_failure tests/macm_controller/test_macm_backoff.py::test_macm_retries_post_start_allocation_oom_without_failure tests/macm_controller/test_macm_backoff.py::test_macm_retries_steady_state_oom_without_failure -q`
  passed with `4 passed in 1.29s`, closing the final local-review residual
  test gap for recoverable OOM retry behavior.
- Final focused runtime-health plus OOM tests:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_runtime_error_as_failure tests/cuda_controller/test_throttle.py::test_cuda_records_post_start_allocation_runtime_error_as_failure tests/cuda_controller/test_throttle.py::test_cuda_retries_post_start_allocation_oom_without_failure tests/cuda_controller/test_throttle.py::test_cuda_retries_steady_state_oom_without_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_records_post_start_allocation_runtime_error_as_failure tests/macm_controller/test_macm_backoff.py::test_macm_retries_post_start_allocation_oom_without_failure tests/macm_controller/test_macm_backoff.py::test_macm_retries_steady_state_oom_without_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_cuda_mps_worker tests/global_controller/global_keep_test.py::test_global_runtime_error_preserves_rank_prefixed_child_failure -q`
  passed with `13 passed in 1.30s`.
- Final broader controller/status tests after OOM coverage:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/macm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  passed with `58 passed, 10 skipped in 1.25s`.
- Final full tests after OOM coverage:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `558 passed, 11 skipped`.
