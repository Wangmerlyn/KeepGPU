# Reject Empty CLI GPU IDs Plan

## Root Cause

`src/keep_gpu/cli.py` treats any falsy `gpu_ids` value as omitted. That makes an explicit empty CLI value such as `--gpu-ids ""` resolve to `None`, which means all visible GPUs.

## Goal

Keep omitted `--gpu-ids` as "all visible GPUs", but reject explicit empty or whitespace-only values before service auto-start/RPC or blocking controller creation.

## Scope

- Update CLI parsing in `src/keep_gpu/cli.py`.
- Add CLI regression tests for service and blocking mode.
- Document the public contract in CLI docs, README, and `AGENTS.md`.

## Tasks

- [x] Add failing tests for `keep-gpu start --gpu-ids ""` and whitespace-only values rejecting before service startup/RPC.
- [x] Add a failing blocking-mode test for `keep-gpu --gpu-ids ""` rejecting before `_run_blocking`.
- [x] Add or update direct `_parse_gpu_ids` coverage for omitted versus explicit empty values.
- [x] Implement the minimal parser change that distinguishes `None` from explicit empty/whitespace strings.
- [x] Update user-facing docs and repository guidance.
- [x] Run targeted tests and `git diff --check`.
- [x] Commit with `fix(cli): reject empty gpu id selections`.

## Verification Checklist

- [x] RED: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q` failed for the new empty-selection tests before implementation.
- [x] GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q` passed after implementation.
- [x] `git diff --check` passes.
