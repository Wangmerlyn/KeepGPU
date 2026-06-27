# Telemetry-Unknown Backoff Fix Plan

## Background

KeepGPU advertises polite, low-power keepalive behavior through `busy_threshold`,
but controller behavior currently fails open when utilization telemetry is
unavailable. CUDA maps unknown NVML utilization to `0%`, ROCm runs batches when
`rocm-smi` is unavailable, and MPS ignores `busy_threshold` entirely.

## Goal

Make telemetry-unavailable cycles conservative by default: if
`busy_threshold >= 0` and utilization cannot be measured, skip keepalive compute
for that cycle. Preserve `busy_threshold=-1` as the explicit unconditional
keepalive mode.

## Design

- CUDA: keep `_monitor_utilization()` as the telemetry boundary, but return
  `None` when utilization is unavailable. Update `_should_run_batch()` to accept
  `Optional[int]` and return `False` for unknown utilization unless
  `busy_threshold < 0`.
- ROCm: add the same decision helper and route `_query_utilization()` through
  it so missing `rocm-smi` sleeps instead of running.
- MPS: because there is no utilization telemetry, run compute only when
  `busy_threshold < 0`; otherwise reserve memory and sleep between cycles.
- Docs: update AGENTS and user docs so agents/users know that unknown telemetry
  is eco-safe and `-1` is the opt-in unconditional mode.

## Tasks

- [x] Add no-GPU tests for CUDA decision behavior:
  - unknown utilization backs off when threshold is non-negative,
  - unknown utilization runs only when threshold is `-1`,
  - `_monitor_utilization()` preserves `None` instead of converting it to `0`.
- [x] Add no-GPU tests for ROCm decision behavior:
  - unknown utilization backs off when threshold is non-negative,
  - unknown utilization runs only when threshold is `-1`.
- [x] Add no-GPU tests for MPS decision behavior:
  - MPS does not run batches with non-negative threshold because telemetry is
    unavailable,
  - MPS runs batches when threshold is `-1`.
- [x] Implement the minimal controller changes.
- [x] Update `AGENTS.md`, README/docs guidance, and plan verification notes.
- [x] Run targeted controller tests, full tests, docs build, and pre-commit.
  - `PYTHONPATH=$PWD/src pytest tests/cuda_controller/test_throttle.py tests/rocm_controller tests/macm_controller tests/single_gpu_controller/test_release_contract.py -q`: 12 passed, 6 skipped.
  - `PYTHONPATH=$PWD/src pytest tests -q`: 81 passed, 12 skipped.
  - `PYTHONPATH=$PWD/src mkdocs build`: passed with existing Material warning and unnav'd docs notices.
  - `pre-commit run --all-files`: passed after Black reformatted controller files.
- [ ] Open PR, run local subagent review, resolve all comments, and squash
  merge only when GitHub checks and local review are clean.
