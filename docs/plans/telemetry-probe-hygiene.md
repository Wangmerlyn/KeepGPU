# Telemetry Probe Hygiene Fix Plan

## Background

Audit agents found that platform probes can initialize vendor libraries without shutting them down, ROCm detection uses an API name inconsistent with the rest of KeepGPU, mocked NVML telemetry is skipped in no-GPU CI, and Apple Silicon/MPS can be controlled but reports no telemetry.

## Goal

Make hardware probing lifecycle-safe and make telemetry visible on CUDA, ROCm, and MPS without requiring GPUs in CI.

## Solution

- Shut down NVML after CUDA detection probes.
- Use `rsmi_init()` / `rsmi_shut_down()` for ROCm detection, matching telemetry and controller code.
- Run mocked NVML telemetry tests on no-GPU CI.
- Add guarded MPS telemetry that reports one `macm` device with nullable memory fields when MPS counters are unavailable.
- Update docs and AGENTS.md with telemetry/probe expectations.

## Todo

- [x] Add failing tests for NVML probe shutdown, ROCm probe API alignment, no-GPU mocked NVML telemetry, and MPS telemetry.
- [x] Implement platform probe cleanup and ROCm API alignment.
- [x] Implement guarded MPS telemetry in `gpu_info.py`.
- [x] Update docs and AGENTS.md.
- [ ] Run targeted tests, docs build, and pre-commit.
- [ ] Open PR, run local subagent review, resolve all comments, then squash merge only when clean.
