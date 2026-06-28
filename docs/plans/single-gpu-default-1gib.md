# Single-GPU Default 1GiB Plan

## Background

KeepGPU documents omitted public VRAM settings as the low-power `1GiB` default,
and `GlobalGPUController` already follows that contract. Direct Python
single-GPU controllers still defaulted `vram_to_keep` to `"1000 MB"`, creating a
small but visible mismatch for users who instantiate CUDA, ROCm, or Mac M
controllers directly.

## Goal

Make omitted VRAM defaults consistent across `GlobalGPUController`,
`CudaGPUController`, `RocmGPUController`, and `MacMGPUController`.

## Solution

- Extend the Python controller contract test to inspect all four constructor
  signatures for `vram_to_keep="1GiB"`.
- Change only the direct single-GPU controller defaults from `"1000 MB"` to
  `"1GiB"`.
- Document that direct single-GPU Python controllers share the same low-power
  omitted VRAM default.

## Tasks

- [x] Add the failing default-contract test.
- [x] Verify the new test fails on current CUDA, ROCm, and Mac M defaults.
- [x] Update the three direct single-GPU controller defaults.
- [x] Update `AGENTS.md`, API reference, and Python guide wording.
- [x] Run targeted verification and `git diff --check`.

## Validation

- `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py::test_python_controller_default_vram_matches_public_low_power_default -q`
- `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py tests/single_gpu_controller -q`
- `git diff --check`
