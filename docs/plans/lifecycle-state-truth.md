# Lifecycle State Truth Fix Plan

## Background

Audit agents found that KeepGPU can report sessions as stopped before release is proven. `KeepGPUServer.stop_keep()` removes sessions before cleanup completes, `GlobalGPUController.keep()` leaves already-started child controllers running if a later child fails, and release exceptions in child controller threads are currently lost.

## Goal

Make start/stop/release state truthful across the Python controller API, JSON-RPC, REST, CLI JSON output, and dashboard messaging while preserving the existing simple controller flow.

## Design

- Keep `GlobalGPUController` as the orchestrator, but make `keep()` transactional: if one child fails to start, release already-started children in reverse order before re-raising the start failure.
- Make release failures observable: per-device controllers raise `TimeoutError` when their worker thread cannot stop, and `GlobalGPUController.release()` attempts every child release before raising a summarized `RuntimeError`.
- Keep server sessions until release succeeds. On timeout, keep the session visible with `state="stopping"` and `last_error`; on release exception, keep it visible with `state="stop_failed"` and `last_error`.
- Return additive stop result fields: `stopped`, `timed_out`, `failed`, and `errors`. Existing consumers that only read `stopped` continue to work.
- Update dashboard stop messages to reflect timed-out or failed outcomes instead of always saying sessions were released.

## Tasks

- [x] Add failing no-GPU tests for transactional `GlobalGPUController.keep()`.
- [x] Add failing no-GPU tests for summarized `GlobalGPUController.release()` errors.
- [x] Add failing no-GPU tests for CUDA, ROCm, and MPS release timeouts.
- [x] Add failing server tests showing timed-out and failed stops remain visible in session status.
- [x] Add failing REST parity tests for timed-out stop outcomes.
- [x] Add failing dashboard helper tests for stop result messages.
- [x] Add failing repeat-stop tests so duplicate stop requests cannot launch concurrent releases.
- [x] Add failing ROCm release contract test for the no-worker path.
- [x] Add failing dashboard helper test for backend `state="stopping"` after refresh.
- [x] Add failing timeout-race test so a late failure cannot be overwritten by a generic timeout state.
- [x] Implement controller rollback and release error aggregation.
- [x] Implement per-device release timeout exceptions.
- [x] Implement server session state retention and additive stop result fields.
- [x] Implement dashboard stop outcome helper and wire it into session/all-session release messages.
- [x] Make repeated stop requests idempotent while a release is already `stopping`.
- [x] Preserve `stop_failed` if a late release callback wins the timeout race.
- [x] Surface backend lifecycle state and retained release errors in dashboard session cards.
- [x] Update AGENTS.md and docs for lifecycle state semantics.
- [x] Run targeted tests, full tests, dashboard tests/build, docs build, and pre-commit.
  - `PYTHONPATH=$PWD/src pytest tests/global_controller tests/single_gpu_controller tests/mcp tests/test_cli_service_commands.py -q`: 47 passed, 1 skipped.
  - `PYTHONPATH=$PWD/src pytest tests -q`: 76 passed, 12 skipped.
  - `npm test` in `web/dashboard`: 14 passed.
  - `npm run build` in `web/dashboard`: passed and refreshed packaged static dashboard assets.
  - `PYTHONPATH=$PWD/src mkdocs build`: passed with existing Material warning and unnav'd docs notices.
  - `pre-commit run --all-files`: passed after Black reformatted `src/keep_gpu/mcp/server.py`.
- [ ] Open PR, run local subagent review, resolve all comments, and squash merge only when GitHub checks and local review are clean.
