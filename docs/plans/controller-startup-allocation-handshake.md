# Controller Startup Allocation Handshake Plan

## Background

CUDA and ROCm workers currently signal `keep()` startup success after device setup
but before the first permitted keep tensor allocation. If that first allocation
fails with a fatal non-OOM error, service mode can briefly register a false
active session and only later learn about `runtime_failed`.

## Goal

Make CUDA and ROCm `keep()` return success only after startup has either made
real progress safely or intentionally deferred work for eco-safe backoff.

## Design

- Preserve eco-safe behavior: busy or unavailable telemetry with
  `busy_threshold >= 0` may still signal startup because no allocation should run.
- Preserve recoverable OOM behavior: an initial OOM retry is not a fatal startup
  error.
- Treat a first permitted non-OOM allocation failure as startup failure and
  propagate it synchronously, matching the existing MPS controller contract.
- Keep the implementation local to the CUDA/ROCm controller loops; no API change.

## Todo

- [x] Add CUDA `keep()` regression test for first permitted non-OOM allocation
      failure.
- [x] Add ROCm `keep()` regression test for first permitted non-OOM allocation
      failure.
- [x] Add public `keep()` coverage for eco-safe deferral, recoverable OOM retry,
      and bounded ROCm OOM exhaustion.
- [x] Resolve review feedback so failure paths always signal any startup event,
      including internal calls without a `startup_errors` list.
- [x] Resolve review feedback so recoverable CUDA and ROCm startup OOM retries
      clear the backend cache before sleeping and retrying.
- [x] Resolve local review feedback so pre-stopped startup paths still signal
      waiters and ROCm no-error-list setup failures retain failure details.
- [x] Update CUDA and ROCm startup handshakes.
- [x] Update `AGENTS.md` and architecture docs with the explicit first-allocation
      startup contract.
- [x] Run targeted controller tests, full tests, docs build, pre-commit, local
      review, and PR checks before merge.
