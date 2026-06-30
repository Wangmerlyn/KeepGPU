# Telemetry Reinitialization Plan

## Background

KeepGPU keeps runtime utilization eco-safe by treating unavailable telemetry as
busy unless the user explicitly selects unconditional mode. GPU listing helpers
probe vendor libraries independently and then shut them down, which can leave a
long-lived keep-loop telemetry object with stale initialization state.

## Goal

Recover CUDA NVML and ROCm SMI utilization after an external probe shuts down the
vendor library, without expanding the public API or adding background polling.

## Solution

- Add regression tests that simulate vendor shutdown after a successful query.
- Reinitialize the vendor telemetry backend once when a stale-library query
  fails.
- Keep invalid visibility masks and unavailable telemetry as `None`.
- Document the guardrail in `AGENTS.md`.

## Todo

- [x] Add RED tests for CUDA NVML and ROCm SMI stale-library recovery.
- [x] Implement minimal reinitialization logic.
- [x] Update agent guidance for vendor telemetry lifecycle recovery.
- [x] Run targeted tests, pre-commit, and docs build.
- [x] Request local subagent code review before opening the PR.
