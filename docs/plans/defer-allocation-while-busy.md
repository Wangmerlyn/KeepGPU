# Defer Allocation While Busy Plan

## Background

Utilization backoff currently gates only the keep-alive compute batch. CUDA,
ROCm, and MPS controllers still allocate and initialize their keep tensor before
checking whether telemetry says the device is busy or unavailable. That startup
allocation can reserve VRAM and run a random-fill kernel even when the default
eco-safe threshold should make the controller sleep.

## Goal

Check backoff before initial keep tensor allocation, so non-negative
`busy_threshold` values avoid allocation and GPU work while the device is busy
or telemetry is unavailable. `busy_threshold=-1` remains the explicit
unconditional allocation/compute mode.

## Solution

- Add RED no-GPU tests proving CUDA, ROCm, and MPS do not call `torch.rand`
  before backoff allows allocation.
- Add no-GPU positive-path tests proving CUDA/ROCm allocate after telemetry
  becomes idle, and that `busy_threshold=-1` still allocates unconditionally.
- Add ROCm coverage proving busy deferrals do not consume allocation retries.
- Move each controller's initial allocation attempt behind its existing
  `_should_run_batch()` decision.
- Keep retry behavior simple: if telemetry says busy/unavailable, wait one
  interval and try the backoff decision again.
- Update architecture and user docs to state that startup allocation is also
  deferred by non-negative utilization backoff.

## Tasks

- [x] Add RED allocation-before-backoff tests for CUDA, ROCm, and MPS.
- [x] Implement allocation deferral in each controller.
- [x] Add positive-path coverage for idle-after-busy and unconditional
      allocation modes.
- [x] Add ROCm retry accounting coverage for busy deferrals.
- [x] Update `AGENTS.md`, architecture/docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

Baseline:

- `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py tests/rocm_controller/test_rocm_backoff.py tests/macm_controller/test_macm_backoff.py tests/global_controller/global_keep_test.py tests/single_gpu_controller/test_release_contract.py -q`:
  `18 passed, 2 skipped`.

Completed:

- RED focused regression:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py::test_cuda_busy_utilization_defers_initial_allocation tests/rocm_controller/test_rocm_backoff.py::test_rocm_busy_utilization_defers_initial_allocation tests/macm_controller/test_macm_backoff.py::test_macm_unavailable_utilization_defers_initial_allocation -q`
  failed with all three tests hitting `allocation should wait for idle telemetry`.
- GREEN focused regression: same command, `3 passed`.
- `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py tests/rocm_controller/test_rocm_backoff.py tests/macm_controller/test_macm_backoff.py -q`:
  `17 passed, 1 skipped`.
- `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py tests/rocm_controller/test_rocm_backoff.py tests/macm_controller/test_macm_backoff.py tests/global_controller/global_keep_test.py tests/single_gpu_controller/test_release_contract.py -q`:
  `27 passed, 2 skipped`.
- `PYTHONPATH=$PWD/src pytest tests -q`: `263 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  version warning and docs-nav notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check`: passed.
- Local subagent code review: passed. Follow-up review confirmed the CLI docs
  parity fix and ROCm retry regression, with no remaining Critical or Important
  blockers.
