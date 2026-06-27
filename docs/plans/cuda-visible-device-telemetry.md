# CUDA Visible-Device Telemetry Plan

## Background

CUDA controllers use visible CUDA ordinals such as `cuda:0`, while NVML indexes
physical devices. When `CUDA_VISIBLE_DEVICES=3,5`, visible CUDA rank `1` refers
to physical GPU `5`, but the current NVML monitor queries physical index `1`.
That can make eco backoff decisions from the wrong GPU and run keepalive compute
on a busy device.

## Goal

Make CUDA utilization backoff read telemetry for the same device that the CUDA
controller is keeping, preserving conservative sleep-only behavior when the
visible-rank to NVML-handle mapping cannot be determined.

## Design

- Keep `CudaGPUController` passing its visible CUDA rank to the monitor.
- Teach `NVMLMonitor` to resolve that visible rank through
  `CUDA_VISIBLE_DEVICES` before querying NVML.
- If `CUDA_VISIBLE_DEVICES` is unset, keep the current direct index behavior.
- If `CUDA_VISIBLE_DEVICES` contains numeric device IDs, map visible rank to the
  corresponding physical NVML index.
- If it contains UUID tokens, use `nvmlDeviceGetHandleByUUID` when available.
- If the token is missing, out of range, empty, or unsupported, return `None`
  rather than querying a possibly wrong physical GPU. Non-negative
  `busy_threshold` already treats `None` as a conservative sleep cycle.
- Update docs and `AGENTS.md` so contributors preserve this visible-rank
  mapping when changing CUDA telemetry.

## Todo

- [x] Add failing monitor tests for `CUDA_VISIBLE_DEVICES=3,5` proving visible
      rank `1` queries physical index `5`.
- [x] Add failing tests for out-of-range, invalid, and unsupported UUID tokens
      returning `None` instead of falling back to the visible ordinal.
- [x] Add failing tests for invalid or unsupported tokens before the requested
      visible ordinal returning `None` for later ordinals.
- [x] Add UUID-token coverage for NVML modules that support
      `nvmlDeviceGetHandleByUUID`.
- [x] Implement visible CUDA rank to NVML handle resolution in
      `src/keep_gpu/utilities/gpu_monitor.py`.
- [x] Update `AGENTS.md`, README, and Python/CLI docs with the CUDA telemetry
      mapping behavior.
- [x] Run targeted GPU monitor/CUDA tests, full tests, docs build,
      pre-commit, and local subagent review.
- [ ] Open the PR, wait for PR checks/reviews, resolve every required comment,
      squash merge, and clean up the worktree.
