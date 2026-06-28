# Duplicate CUDA Mask Telemetry Plan

## Background

CUDA controllers use visible ranks such as `cuda:0`, while NVML queries physical
devices. `gpu_monitor.py` resolves `CUDA_VISIBLE_DEVICES` so utilization
backoff reads telemetry for the same device that the controller keeps. However,
a duplicate mask such as `CUDA_VISIBLE_DEVICES=0,0` allowed visible rank `1` to
query the same physical GPU as rank `0`.

`gpu_info.py` already treats duplicate CUDA masks as invalid and hides the GPU
list rather than guessing. Telemetry should follow the same eco-safe rule:
ambiguous mappings return unavailable utilization so non-negative
`busy_threshold` values sleep instead of making a decision from aliased data.

## Goal

Make duplicate CUDA visibility masks resolve to unavailable utilization in
`NVMLMonitor`, including equivalent numeric spellings such as `0,00`, without
changing normal numeric/UUID mask handling or existing invalid-token behavior.

## Solution

- Add a no-GPU regression test with fake NVML showing
  `CUDA_VISIBLE_DEVICES=0,0` returns `None` for visible rank `1`.
- Add follow-up regressions for equivalent numeric aliases such as `0,00` and
  exact duplicate UUID tokens.
- Reject duplicate `CUDA_VISIBLE_DEVICES` tokens before NVML handle lookup in
  `gpu_monitor.py`, normalizing numeric tokens before comparison.
- Update user docs and `AGENTS.md` to state that duplicate CUDA masks are
  ambiguous and report unavailable telemetry.

## Tasks

- [x] Add RED duplicate-mask telemetry test.
- [x] Add review-driven regressions for numeric aliases and duplicate UUID
      tokens.
- [x] Implement normalized duplicate-token rejection in `NVMLMonitor`.
- [x] Update `AGENTS.md`, README, architecture/API/CLI/MCP/Python docs, and
      this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

Baseline:

- `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py -q`:
  `15 passed`.

Completed:

- RED focused regression:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_for_duplicate_cuda_visible_devices -q`
  failed because utilization returned `99` instead of `None`.
- GREEN monitor shard:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py -q`:
  `18 passed`.
- Review-driven focused regression:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_for_equivalent_numeric_cuda_visible_devices tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_for_duplicate_uuid_cuda_visible_devices -q`
  failed before numeric token normalization because `0,00` returned `99`
  instead of `None`.
- `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py tests/utilities/test_gpu_info.py tests/utilities/test_platform_manager.py -q`:
  `42 passed, 1 skipped`.
- `PYTHONPATH=$PWD/src pytest tests -q`: `266 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  version warning and docs-nav notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check && git diff --cached --check`: passed.
- Local subagent code review: initial review found numeric aliases such as
  `0,00` still aliased to the same physical GPU; follow-up fix pending review.
