# ROCm Runtime Failure Retention Plan

## Background

CUDA and MPS controllers retain unexpected post-start worker failures in
`_failure_exc`, which lets `GlobalGPUController.runtime_error()` and the service
status path report `state="runtime_failed"` instead of leaving a dead or broken
worker looking active.

ROCm has the same health hook shape, but its loop still treats non-OOM
post-start `RuntimeError`s and unexpected worker exceptions as sleep-and-retry
conditions. A dead ROCm worker with `_failure_exc` is also not cleaned by
`release()` unless the stop event was already set.

## Goal

Make ROCm runtime-failure behavior match the shared controller contract: OOM
allocation/backoff remains retryable, while non-OOM worker failures are retained
and dead failed workers release cache/state cleanly.

## Solution

- Add RED coverage for ROCm non-OOM allocation and steady-state worker failures.
- Add RED release-contract coverage for dead ROCm workers with retained failures.
- Update the ROCm worker loop to record unexpected fatal failures in
  `_failure_exc` and return, preserving OOM retry behavior.
- Update ROCm `release()` to clear cache/state when a retained failed worker is
  already dead, matching CUDA/MPS.
- Update `AGENTS.md` so future controller work treats ROCm as part of the shared
  runtime-failure contract.

## Tasks

- [x] Write failing ROCm tests for non-OOM allocation failure, steady-state
      `RuntimeError`, and unexpected worker exception retention.
- [x] Write failing release-contract test for dead runtime-failed ROCm worker
      cleanup.
- [x] Implement the minimal ROCm controller changes.
- [x] Update `AGENTS.md` and this plan with verification evidence.
- [x] Run targeted ROCm/controller tests.
- [x] Run broader tests, docs build, pre-commit, and whitespace checks.
- [x] Request local subagent code review and resolve findings before PR.
- [ ] Open PR, resolve hosted comments/checks, squash merge, and clean the
      worktree/branches.

## Verification Log

- Baseline before edits:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  passed with 56 passed, 1 skipped.
- RED after adding regression tests:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_unexpected_post_start_worker_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_runtime_error_as_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_allocation_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_worker -q`
  failed with five ROCm-specific failures: fatal worker paths called
  `stop_evt.wait()` instead of retaining `_failure_exc`, and dead failed ROCm
  release skipped cache cleanup.
- GREEN after implementation:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_unexpected_post_start_worker_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_runtime_error_as_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_allocation_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_worker -q`
  passed with 9 passed.
- Local subagent reviewers found one remaining gap: non-`RuntimeError`
  exceptions in ROCm's post-start initial allocation loop could still exit the
  worker without `_failure_exc`. Added
  `test_rocm_records_unexpected_post_start_allocation_failure` and the matching
  fatal-retention branch. The repaired focused command
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_unexpected_post_start_allocation_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_unexpected_post_start_worker_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_runtime_error_as_failure tests/rocm_controller/test_rocm_backoff.py::test_rocm_records_post_start_allocation_runtime_error_as_failure tests/single_gpu_controller/test_release_contract.py::test_release_cleans_dead_runtime_failed_worker -q`
  passed with 10 passed.
- Focused regression:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller/global_keep_test.py -q`
  passed with 62 passed, 1 skipped after the local review fix.
- Broader affected tests:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/single_gpu_controller/test_release_contract.py tests/global_controller tests/mcp tests/utilities/test_gpu_info.py -q`
  passed with 298 passed, 2 skipped before the review fix.
- Full suite after the local review fix:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 574 passed, 11 skipped.
- Docs/hooks/whitespace after the local review fix:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan notices; `pre-commit run --all-files` passed; `git diff
  --check` passed.
- Local subagent re-review approved after verifying the prior non-`RuntimeError`
  initial-allocation gap was resolved; only the stale plan count above was
  noted and fixed.
