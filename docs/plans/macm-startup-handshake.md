# Mac M Startup Handshake Implementation Plan

**Goal:** Make `MacMGPUController.keep()` fail synchronously when the worker hits a fatal first-allocation startup error, matching the single-GPU startup contract already used by CUDA and ROCm.

**Background:** The Mac M controller currently starts `_keep_loop()` in a daemon thread and immediately logs success. The first MPS allocation happens later in the worker, so an immediate non-OOM allocation failure can be reported only through `allocation_status()` after `keep()` has already returned.

**Approach:** Add a small startup event/error handshake to the Mac M worker path. The worker will signal startup only after it has either completed the first successful allocation, decided to defer allocation because telemetry backoff is active, or captured a fatal startup setup/allocation error. `keep()` will wait for that signal, clean up thread state on fatal startup errors, and raise the captured exception.

**Files:**
- Modify `tests/macm_controller/test_macm_backoff.py` with one regression test for a first `torch.rand` `RuntimeError("mps startup allocation failed")`.
- Modify `src/keep_gpu/single_gpu_controller/macm_gpu_controller.py` to add the startup event/error plumbing.
- Modify `AGENTS.md` to name MPS first-allocation setup failures in the existing synchronous startup-failure contract.

**Todo:**
- [x] Add the failing regression test first.
- [x] Run the new test and record the expected RED failure.
- [x] Implement the minimal Mac M startup handshake.
- [x] Strengthen the regression test to assert startup-failure state cleanup.
- [x] Update `AGENTS.md` with the MPS-specific startup failure contract.
- [x] Run the new test, full Mac M backoff tests, analogous CUDA/ROCm startup tests, and `git diff --check`.
- [x] Commit with `fix(macm): surface startup allocation failures`.
