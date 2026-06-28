# Direct Controller Rank Validation Plan

## Background

Global GPU sessions already validate explicit `gpu_ids` as visible ordinals
after CUDA or ROCm environment filtering. Direct Python users could still
instantiate `CudaGPUController(rank=...)` or `RocmGPUController(rank=...)` with
non-integer, negative, or out-of-range ranks, letting invalid values reach
`torch.device`, backend `set_device`, telemetry, or worker startup.

## Goal

Make direct CUDA/ROCm single-GPU controllers fail fast when `rank` is not a
plain visible device ordinal in the current process environment, while keeping
the existing low-power keep loop and global-controller semantics unchanged.

## Solution

- Add shared `validate_visible_rank(rank, visible_count)` logic to
  `session_config.py`.
- Use it from `CudaGPUController` and `RocmGPUController` constructors after
  constructor-local option validation and before device/thread setup.
- Add no-hardware tests for direct constructor rank validation and utility
  contract coverage.
- Update Python API docs, architecture docs, README, and `AGENTS.md` so future
  changes preserve the visible-rank boundary.

## Tasks

- [x] Add RED direct CUDA/ROCm constructor tests for non-integer and out-of-range
      ranks.
- [x] Implement shared visible-rank validation and wire it into CUDA/ROCm direct
      controllers.
- [x] Update no-GPU CUDA/ROCm worker-startup tests to declare a fake visible
      rank when exercising later startup behavior.
- [x] Add utility-level validation coverage.
- [x] Update `AGENTS.md`, README, API guide, API reference, architecture docs,
      and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and `git diff
      --check`.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py -q -k 'direct_cuda_rocm_controllers_reject'`
  failed before the implementation because direct CUDA/ROCm constructors did
  not reject invalid ranks consistently.
- GREEN focused regression after implementation:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py -q -k 'direct_cuda_rocm_controllers_reject'`
  passed with `10 passed, 31 deselected`.
- Nearby controller shard:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py tests/cuda_controller/test_throttle.py tests/rocm_controller/test_rocm_utilization.py tests/rocm_controller/test_rocm_backoff.py tests/single_gpu_controller/test_release_contract.py -q`
  passed with `71 passed, 1 skipped`.
- Full branch gate after the CUDA no-GPU test harness update:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `396 passed, 11 skipped`.
- Final targeted shard after docs and utility coverage:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py tests/cuda_controller/test_keep_and_release.py tests/cuda_controller/test_throttle.py tests/rocm_controller/test_rocm_utilization.py tests/rocm_controller/test_rocm_backoff.py tests/single_gpu_controller/test_release_contract.py tests/utilities/test_session_config.py -q`
  passed with `120 passed, 3 skipped`.
- Final full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `406 passed, 11 skipped`.
- Docs and hygiene gates:
  `PYTHONPATH=$PWD/src mkdocs build` succeeded with the repository's existing
  Material warning and unnav'd plan notices; `pre-commit run --all-files`
  passed; and `git diff --check` passed.
- Local subagent review before PR: passed with no critical, important, or minor
  findings. The reviewer also ran a focused cache-disabled shard:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src pytest -p no:cacheprovider`
  against the direct-rank, session-config, CUDA startup, and ROCm startup
  shards, reporting `109 passed, 2 skipped`.
- Review follow-up: Gemini requested a clearer `visible_count == 0` message.
  Added a dedicated validator branch and regression so CPU-only/no-visible-GPU
  environments do not report the confusing bound `less than 0`.
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/global_controller/test_contract.py tests/cuda_controller/test_keep_and_release.py tests/rocm_controller/test_rocm_backoff.py tests/rocm_controller/test_rocm_utilization.py -q`
  passed with `110 passed, 2 skipped`; `PYTHONPATH=$PWD/src pytest tests -q`
  passed with `407 passed, 11 skipped`; `PYTHONPATH=$PWD/src mkdocs build`,
  `pre-commit run --all-files`, and `git diff --check` passed after the
  follow-up.
