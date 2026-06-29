# Mac M Input Validation Plan

## Background

`MacMGPUController` checks `rank != 0` and probes
`torch.backends.mps.is_available()` before all cheap public constructor inputs
are validated. On non-MPS hosts, invalid `busy_threshold` values can therefore
surface as a hardware availability error instead of the public validation error.
The rank check also treats `False` as `0`, because `False != 0` is false.

## Goal

Make direct Mac M controller construction fail for invalid public inputs before
any MPS hardware/backend probe, matching the CUDA and ROCm controller contract.

## Solution

- Add no-GPU-safe regression tests showing invalid `busy_threshold`, invalid
  rank types, and out-of-range ranks fail before `torch.backends.mps.is_available`
  is called.
- Validate `busy_threshold`, `iterations`, and `rank` before the MPS
  availability probe.
- Use the shared `validate_visible_rank(rank, 1)` helper for Mac M rank
  validation so `False`, strings, floats, negative values, and non-zero ranks
  use the same public errors as other direct controllers.
- Update API docs and `AGENTS.md` to include the direct Mac M controller in the
  cheap-input-before-backend-probe contract.

## Tasks

- [x] Add RED constructor validation tests under `tests/macm_controller/`.
- [x] Implement centralized Mac M constructor validation before MPS probing.
- [x] Update `AGENTS.md`, API docs, and this plan.
- [x] Run targeted tests required for this validation fix.
- [x] Run full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent code review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge,
      and clean the worktree.

## Verification

- RED: `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py -q -k 'constructor_rejects'`
  failed with 6 failures on the old implementation. Invalid
  `busy_threshold=101` and `rank=False` reached the forbidden MPS availability
  probe; invalid rank values `"0"`, `1.5`, `-1`, and `1` raised the old
  Mac-specific rank error instead of the shared rank validation contract.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py -q -k 'constructor_rejects'`
  passed: 6 passed, 11 deselected in 1.46s.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py tests/macm_controller/test_macm_basic.py tests/global_controller/test_contract.py -q`
  passed: 59 passed, 4 skipped in 1.68s.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py tests/global_controller/test_contract.py -q`
  passed: 59 passed in 1.44s.
- GREEN after integration:
  `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py -q -k 'constructor_rejects or preserves_unavailable_mps_runtime_error'`
  passed with 7 passed, 11 deselected.
- GREEN after integration:
  `PYTHONPATH=$PWD/src pytest tests/macm_controller/test_macm_backoff.py tests/macm_controller/test_macm_basic.py tests/global_controller/test_contract.py -q`
  passed with 59 passed, 4 skipped.
- FULL: `PYTHONPATH=$PWD/src pytest tests -q` passed with 604 passed,
  11 skipped.
- DOCS: `PYTHONPATH=$PWD/src mkdocs build` passed with the repository's
  existing Material warning and unnav'd plan notices.
- HYGIENE: `pre-commit run --all-files` passed.
- CHECK: `git diff --check` passed with no output.
- REVIEW: local subagent review found no critical, important, or minor issues
  and marked the branch ready to merge.
