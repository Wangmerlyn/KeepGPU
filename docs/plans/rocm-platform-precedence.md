# ROCm Platform Precedence Plan

## Background

ROCm PyTorch builds can report `torch.cuda.is_available()` while also exposing a
truthy `torch.version.hip`. KeepGPU already avoided the torch-CUDA path in that
case, but `_check_cuda()` still fell through to NVML. On a mixed host where NVML
initializes, platform detection selected CUDA before ROCm and cached the wrong
controller family.

## Goal

Prefer the active HIP/ROCm PyTorch runtime over NVML-based CUDA fallback so
KeepGPU selects ROCm controllers on ROCm torch builds.

## Solution

- Add a no-GPU regression test for `get_platform()` with HIP torch and working
  NVML.
- Update `_check_cuda()` to return `False` immediately when `torch.version.hip`
  is truthy, skipping the NVML fallback.
- Keep a direct `_check_cuda()` test proving HIP builds do not probe NVML.
- Document the detection invariant in `AGENTS.md` and the architecture guide.

## Todo

- [x] Run the platform-manager baseline.
- [x] Add the failing ROCm-over-NVML platform detection test.
- [x] Verify the new test fails on current behavior.
- [x] Implement the minimal CUDA probe guard.
- [x] Update stale platform-manager test expectations.
- [x] Run the platform-manager shard.
- [x] Run broader tests, docs build, pre-commit, and local subagent review.
- [ ] Open a PR, resolve review comments, squash merge, and clean up the branch.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/utilities/test_platform_manager.py -q`
- `PYTHONPATH=$PWD/src pytest tests/utilities -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
