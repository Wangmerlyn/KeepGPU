# Mac M Startup Stop Failure Retention Plan

## Background

`MacMGPUController._keep_loop()` could reach its `tensor is None` exit when the
stop event was already set before the first allocation. That path logged a
generic allocation message and returned without signaling `startup_evt` or
retaining a failure in `allocation_status()`. CUDA and ROCm already report this
as a concrete `stopped before ... startup allocation` lifecycle cause.

## Goal

Keep MPS startup lifecycle state truthful when a startup is stopped before first
allocation: signal startup completion and preserve the specific failure cause so
service status does not degrade to a timeout or lose the reason.

## Solution

- Add a Mac M regression test mirroring the CUDA/ROCm startup-stop contract.
- When MPS exits before first allocation and startup has not been signaled,
  record `rank N: stopped before MPS startup allocation` in `startup_errors` or
  `_failure_exc`, then set `startup_evt`.
- Leave post-confirmation exits as quiet debug exits so eco-safe deferred or
  recoverable paths keep their current behavior.
- Document the MPS-specific lifecycle invariant in `AGENTS.md`.

## Verification

- RED:
  `PYTHONPATH=src pytest tests/macm_controller/test_macm_backoff.py::test_macm_preserves_failure_when_stopped_before_startup_allocation -q`
  failed because `startup_evt` was not set.
- GREEN:
  `PYTHONPATH=src pytest tests/macm_controller/test_macm_backoff.py::test_macm_preserves_failure_when_stopped_before_startup_allocation -q`
  passed.
- Mac M slice:
  `PYTHONPATH=src pytest tests/macm_controller/test_macm_backoff.py -q` passed
  with 21 tests.

## Remaining Checks

- [x] Run the targeted controller suite.
- [x] Run the full test suite.
- [x] Run `mkdocs build --strict`.
- [x] Run `pre-commit run --all-files --show-diff-on-failure`.
- [x] Run local subagent code review before PR.
