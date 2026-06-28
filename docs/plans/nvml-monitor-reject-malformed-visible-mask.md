# NVML Monitor Reject Malformed Visible Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CUDA NVML utilization telemetry fail closed for malformed `CUDA_VISIBLE_DEVICES` masks so eco-safe backoff treats telemetry as unavailable.

**Architecture:** Keep the change inside `src/keep_gpu/utilities/gpu_monitor.py`. Parse the full CUDA visibility mask before any NVML handle lookup, align malformed-mask behavior with `gpu_info.py`, and preserve existing valid numeric and UUID mapping behavior.

**Tech Stack:** Python, pytest, NVML through the `nvidia-ml-py` `pynvml` module.

---

## Root Cause

`NVMLMonitor._get_handle_for_visible_index()` currently splits `CUDA_VISIBLE_DEVICES`, drops empty tokens while checking duplicates, then resolves only tokens up to the requested visible rank. A malformed mask such as `0,,2` can therefore let `get_gpu_utilization(0)` query physical GPU `0` before the empty token is noticed. That conflicts with `gpu_info.py`, which treats malformed CUDA masks as no visible GPUs.

For eco-safe controller behavior, untrusted telemetry must be unavailable (`None`) so non-negative `busy_threshold` values back off instead of allocating keep tensors from partial telemetry.

## Scope

- Modify only the NVML monitor visibility-mask handling in `src/keep_gpu/utilities/gpu_monitor.py`.
- Update regression coverage in `tests/utilities/test_gpu_monitor.py`.
- Update project guidance in `AGENTS.md` and architecture/user docs only where needed.
- Do not change controller startup, GPU listing, ROCm visibility, or MCP/REST behavior.

## Tasks

- [x] Create this plan with root cause, scope, tasks, and verification checklist.
- [x] Update tests that previously allowed partial handle queries for malformed CUDA masks:
  - `CUDA_VISIBLE_DEVICES="0,,"`
  - `CUDA_VISIBLE_DEVICES="0,,2"`
  - `CUDA_VISIBLE_DEVICES="3,,5"`
- [x] Add coverage for `CUDA_VISIBLE_DEVICES="-1,0"` returning `None` without NVML handle or UUID lookup.
- [x] Run targeted tests before production code and record the RED result.
  - Command: `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py -q`
  - Result before production change: 4 failed, 17 passed. Failures showed `0,,` and `0,,2` still returned partial telemetry, while `0,-1,2` and `3,,5` still queried an earlier physical index.
- [x] Implement the minimal monitor fix by validating the full mask before any handle lookup.
- [x] Update docs to state CUDA telemetry mask parsing fails closed for malformed masks.
- [x] Run targeted verification:
  - `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py tests/utilities/test_gpu_info.py -q`
  - `git diff --check`
- [x] Commit with `fix(telemetry): reject malformed cuda visibility masks`.
- [x] Address local review follow-up:
  - Add RED coverage for `CUDA_VISIBLE_DEVICES="0,99,2"` queried at visible rank
    `0`, proving a later out-of-range numeric token could still return partial
    rank-0 telemetry.
  - Validate all numeric CUDA mask tokens against the NVML device count before
    any handle lookup.
  - Sync Python, MCP/REST, CLI reference, API reference, README, architecture,
    and `AGENTS.md` wording around fail-closed CUDA telemetry masks.
- [x] Address final local review follow-up:
  - Add RED coverage for `CUDA_VISIBLE_DEVICES="0,GPU-typo"` queried at visible
    rank `0`, proving a later unresolved UUID token could still return partial
    numeric rank-0 telemetry.
  - Resolve the full visibility mask before returning any requested rank handle,
    with UUID tokens checked before numeric handle queries so unresolved UUIDs
    cannot permit partial numeric telemetry.
- [x] Address CodeRabbit mixed-alias follow-up:
  - Add RED coverage for `CUDA_VISIBLE_DEVICES="0,GPU-zero"` where the UUID alias
    resolves to the same physical GPU as numeric token `0`.
  - De-duplicate resolved handles by canonical NVML device index when available,
    returning unavailable telemetry when two visible tokens alias the same GPU.

## Verification Checklist

- [x] RED: Targeted GPU monitor tests fail because malformed masks now expect no NVML handle lookup.
- [x] GREEN: `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py tests/utilities/test_gpu_info.py -q` passes with 41 passed, 1 skipped.
- [x] Clean diff check: `git diff --check` passes.
- [x] Git status contains only scoped files before commit.
- [x] Review RED: `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_when_prior_numeric_index_is_invalid -q` failed with `assert 41 is None` before the count-based fix.
- [x] Final review RED: `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_when_later_uuid_token_is_unresolved -q` failed with `assert 41 is None` before the full-mask resolver.
- [x] CodeRabbit RED: `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_monitor.py::test_monitor_returns_none_for_numeric_uuid_alias_to_same_device -q` failed with `assert 50 is None` before resolved-handle de-duplication.
