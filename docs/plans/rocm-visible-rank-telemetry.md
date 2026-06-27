# ROCm Visible-Rank Telemetry Plan

## Background

Public GPU selection uses visible ordinals after user-supplied visibility masks,
but ROCm telemetry currently sends those visible ranks directly to
`rocm_smi.rsmi_dev_busy_percent_get()`. With masks such as
`HIP_VISIBLE_DEVICES=3,5`, visible rank `1` refers to physical GPU `5`, while
the current code queries SMI index `1`.

## Goal

Keep ROCm keepalive backoff and GPU listing telemetry aligned with the same
visible ordinals users pass to KeepGPU. If the visible-to-physical mapping is
ambiguous, return unavailable utilization instead of querying a possibly wrong
device.

## Solution

- Add a small ROCm visibility resolver in `src/keep_gpu/utilities/` shared by
  the ROCm controller and `gpu_info`.
- Resolve `ROCR_VISIBLE_DEVICES` as the base mask, then apply one
  `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` overlay. If both HIP and CUDA
  masks are set, they must normalize to the same numeric token list.
- Accept only unique non-negative numeric tokens. Empty, `-1`, malformed,
  duplicate, unsupported, out-of-range, or conflicting overlay masks resolve to
  unavailable telemetry.
- Use the resolved physical SMI index for `rsmi_dev_busy_percent_get()`;
  otherwise return `None` so non-negative `busy_threshold` sleeps.
- Keep GPU listing IDs as visible ordinals and add `physical_id` only when the
  ROCm SMI index is known.

## Tasks

- [x] Add failing no-hardware tests for `RocmGPUController._query_utilization()`
      mapping visible rank `1` through `HIP_VISIBLE_DEVICES=3,5` to SMI index
      `5`.
- [x] Add failing no-hardware tests that malformed or conflicting ROCm
      visibility masks return `None` and do not call SMI.
- [x] Add failing no-hardware `gpu_info` tests that list visible IDs while
      querying resolved physical SMI indexes and exposing `physical_id`.
- [x] Implement the shared resolver and route ROCm controller/listing telemetry
      through it.
- [x] Update `AGENTS.md`, README, CLI/Python/API docs, and this plan's
      verification notes.
- [x] Run targeted tests, full tests, docs build, and pre-commit.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- Red checks:
  - Initial test setup was corrected after `vram_to_keep=1` proved invalid.
  - Final RED run failed as expected: current code returned visible-index
    utilization (`40`/`41`) and omitted ROCm `physical_id`.
- Green check completed:
  - `PYTHONPATH=$PWD/src pytest tests/rocm_controller/test_rocm_utilization.py tests/utilities/test_gpu_info.py -q`
  - Result: `24 passed, 1 skipped`.
- Completed checks:
  - `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/utilities/test_gpu_info.py -q`
    - Result: `26 passed, 1 skipped`.
  - `PYTHONPATH=$PWD/src pytest tests -q`
    - Result: `235 passed, 11 skipped`.
  - `PYTHONPATH=$PWD/src mkdocs build`
    - Result: passed with the existing Material warning and unlisted plan-page
      notices.
  - `pre-commit run --all-files`
    - Result: passed.
  - `git diff --check`
    - Result: passed.
- Local subagent review:
  - Initial review found the new resolver/plan were untracked and one stale
    Python guide note still described CUDA-only fallback.
  - Follow-up review after staging the new files and updating the Python guide
    returned no remaining must-fix findings.
