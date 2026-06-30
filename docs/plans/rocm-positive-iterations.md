# ROCm and Shared Positive Iterations Plan

## Background

`RocmGPUController` accepted `iterations=0` and negative values, while CUDA and
Mac M controllers already reject non-positive keep workload iteration counts.
Because ROCm batches run `range(self.iterations)`, non-positive values can turn a
keep batch into a silent no-op while still synchronizing and logging. External
PR review also noted that non-integer iteration counts can pass construction and
fail later in the background keep loop.

The final fix centralizes the positive-integer workload iteration rule for all
single-GPU controllers so CUDA, ROCm, and Mac M share the same fail-fast public
contract.

## Goal

Keep ROCm behavior aligned with the other single-GPU controllers and ensure all
platform backends reject invalid workload iteration counts before keep workers
start.

## Solution

- Add a no-GPU regression test for `iterations=0` and `iterations=-1`.
- Validate positive integer workload iteration counts before constructors start
  keep workers.
- Apply the shared validation helper to CUDA `relu_iterations`, ROCm
  `iterations`, and Mac M `iterations`. The historical CUDA
  `matmul_iterations` alias has since been removed; do not reintroduce it as
  part of iteration validation work.
- Document that keep workload iteration counts must be positive integers, and
  preserve the eco-safe expectation that KeepGPU should never silently run
  no-op sessions.

## Todo

- [x] Run the ROCm controller baseline.
- [x] Add the failing ROCm constructor validation test.
- [x] Verify the new test fails on current behavior.
- [x] Implement the minimal constructor validation.
- [x] Verify the focused test and ROCm shard pass.
- [x] Run broader tests, docs build, pre-commit, and local subagent review.
- [x] Address PR review by adding positive-integer workload iteration validation
      across single-GPU controllers.
- [ ] Open a PR, resolve review comments, squash merge, and clean up the branch.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/rocm_controller -q`
- `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/single_gpu_controller/test_release_contract.py -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
