# ROCm Max Allocation Retries Validation Plan

## Background

`RocmGPUController(max_allocation_retries=...)` currently stores caller input directly. Non-integer, boolean, zero, or negative values can make the allocation retry loop crash or behave ambiguously after `keep()` has already returned.

## Goal

Validate `max_allocation_retries` at `RocmGPUController` construction time so callers get a synchronous error before any worker startup.

## Solution

Reuse the existing plain positive integer validation style in `src/keep_gpu/utilities/session_config.py`. Accept only `None` or a positive plain `int`; reject `bool`, strings, zero, and negative values.

## Tasks

- [x] Add RED tests in `tests/rocm_controller/test_rocm_backoff.py` for invalid values `"1"`, `True`, `0`, and `-1`, plus a valid positive integer.
- [x] Run the targeted ROCm backoff test and confirm the invalid-value cases fail before implementation.
- [x] Update `src/keep_gpu/single_gpu_controller/rocm_gpu_controller.py` to validate `max_allocation_retries` in `__init__`.
- [x] Add a concise ROCm retry validation note to `AGENTS.md`.
- [x] Run targeted tests and `git diff --check`.
- [x] Commit with `fix(rocm): validate allocation retry limit`.

## Verification Notes

Required targeted checks:

- `pytest tests/rocm_controller/test_rocm_backoff.py -q`
- `pytest tests/global_controller/test_contract.py tests/rocm_controller/test_rocm_backoff.py tests/utilities/test_session_config.py -q`
- `git diff --check`
