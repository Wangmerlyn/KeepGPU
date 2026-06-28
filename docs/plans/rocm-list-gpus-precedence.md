# ROCm GPU Listing Precedence Plan

## Background

Platform detection already treats a PyTorch build with truthy `torch.version.hip`
as ROCm before NVML-based CUDA fallback. `get_gpu_info()`, however, still tries
NVML first for `list_gpus`/`/api/gpus`. On mixed hosts with ROCm torch and
working NVML, GPU listing can expose CUDA/NVML records instead of ROCm-visible
ordinals and ROCm SMI metadata.

## Goal

Keep GPU listing aligned with the active HIP/ROCm runtime: when
`torch.version.hip` is truthy, `get_gpu_info()` should prefer ROCm telemetry and
skip NVML CUDA listing, using torch's HIP-backed listing when ROCm SMI is
unavailable.

## Solution

- Add a RED GPU-info regression with HIP torch, working ROCm SMI, and working
  NVML; it must return ROCm records and avoid NVML queries.
- Teach `get_gpu_info()` to try ROCm SMI, then torch's HIP-backed listing, when
  the active torch runtime is HIP/ROCm.
- Preserve existing CUDA/NVML-first behavior for non-HIP torch builds.
- Update agent/docs guidance to keep `list_gpus` platform precedence aligned
  with controller platform detection.

## Tasks

- [x] Run focused telemetry baseline.
- [x] Add RED HIP-over-NVML GPU listing regressions.
- [x] Implement minimal `get_gpu_info()` precedence fix.
- [x] Update `AGENTS.md`, docs, and this plan.
- [x] Run focused telemetry tests, full tests, docs build, and pre-commit.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge,
      and clean the worktree.

## Verification

Completed so far:

- Focused telemetry baseline:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py tests/utilities/test_platform_manager.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py -q`,
  `54 passed, 1 skipped`.
- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py::test_get_gpu_info_prefers_rocm_when_hip_torch_and_nvml_are_both_available -q`,
  `1 failed` because `get_gpu_info()` returned NVML CUDA records before ROCm.
- RED ROCm-SMI-unavailable regression:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py::test_get_gpu_info_hip_torch_skips_nvml_when_rocm_smi_unavailable -q`,
  `1 failed` because HIP torch still fell through to NVML CUDA records.
- GREEN regression:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py::test_get_gpu_info_prefers_rocm_when_hip_torch_and_nvml_are_both_available tests/utilities/test_gpu_info.py::test_get_gpu_info_hip_torch_skips_nvml_when_rocm_smi_unavailable -q`,
  `2 passed`.
- GPU-info shard:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py -q`,
  `20 passed, 1 skipped` after adding the review-followup missing
  `torch.version` regression.
- Telemetry shard:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py tests/utilities/test_platform_manager.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py -q`,
  `58 passed, 1 skipped` after the review-followup regression.
- Full test suite:
  `PYTHONPATH=$PWD/src pytest tests -q`, `321 passed, 11 skipped`.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build`, passed with the known
  Material/MkDocs warning and unnav'd plan notices.
- Pre-commit:
  `pre-commit run --all-files`, passed.
- Review follow-up:
  Gemini suggested guarding missing `torch.version`; updated the HIP check to
  use nested `getattr` and added
  `test_get_gpu_info_allows_torch_without_version_attribute`.
- Local subagent review:
  code-quality reviewer reported no findings. Spec reviewer reported no
  blockers but noted `_query_torch()` still read `torch.version.hip` directly
  on fallback; added a RED regression
  `test_get_gpu_info_torch_fallback_allows_missing_version_attribute`, observed
  `1 failed` because the torch fallback returned no GPU records, then guarded
  `_query_torch()` with nested `getattr` and confirmed the targeted follow-up
  group passed with `4 passed`.
