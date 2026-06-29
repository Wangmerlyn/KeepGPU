# Rank Type Before Device Count Plan

## Background

Direct CUDA and ROCm controllers call
`validate_visible_rank(rank, torch.cuda.device_count())`. Python evaluates
`torch.cuda.device_count()` before `validate_visible_rank()` can reject
non-integer public ranks. If the CUDA runtime probe fails, invalid inputs such
as `rank="0"` can be masked by a backend error instead of the shared
`TypeError("rank must be an integer")` contract.

## Goal

Reject non-integer direct-controller ranks before CUDA/ROCm backend device-count
probing, while preserving visible-count checks for plain integer ranks.

## Solution

- Add a shared rank type validator that returns the normalized plain integer
  rank or raises `TypeError("rank must be an integer")`.
- Call that helper in CUDA and ROCm constructors before
  `torch.cuda.device_count()`.
- Keep the existing `validate_visible_rank(rank, visible_count)` range contract
  after the device count is known.
- Update tests and docs so the ordering is explicit.

## Tasks

- [x] Add RED tests for rank type validation before device-count probing.
- [x] Implement shared rank type validation and use it in CUDA/ROCm controllers.
- [x] Update `AGENTS.md`, API docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent code review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge,
      and clean the worktree.

## Verification

- RED utility boundary:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py::test_validate_rank_type_accepts_plain_integer_rank tests/utilities/test_session_config.py::test_validate_rank_type_rejects_non_plain_integer_rank tests/global_controller/test_contract.py::test_direct_cuda_rocm_controllers_reject_non_integer_ranks_before_device_count -q`
  first failed during import because `validate_rank_type` did not exist.
- RED controller ordering:
  after adding the shared helper, the same command failed with six controller
  cases because `torch.cuda.device_count()` still ran before non-integer ranks
  could be rejected.
- GREEN focused rank ordering:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py::test_validate_rank_type_accepts_plain_integer_rank tests/utilities/test_session_config.py::test_validate_rank_type_rejects_non_plain_integer_rank tests/global_controller/test_contract.py::test_direct_cuda_rocm_controllers_reject_non_integer_ranks_before_device_count -q`
  passed with 10 passed.
- GREEN targeted controller/utilities:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/global_controller/test_contract.py tests/cuda_controller tests/rocm_controller -q`
  passed with 143 passed, 5 skipped.
- FULL: `PYTHONPATH=$PWD/src pytest tests -q` passed with 614 passed,
  11 skipped.
- DOCS: `PYTHONPATH=$PWD/src mkdocs build` passed with the repository's
  existing Material warning and unnav'd plan notices.
- HYGIENE: `pre-commit run --all-files` passed.
- CHECK: `git diff --check` passed with no output.
- REVIEW: local subagent review found no critical, important, or minor issues,
  reran `git diff --check`, and marked the branch ready to merge.
