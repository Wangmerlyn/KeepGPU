# Pre-Python 3.10 Annotation Compatibility Plan

## Background

Project formatting/linting now targets Python 3.9, but a few modules use
PEP 604 union annotations without `from __future__ import annotations`. On
Python 3.9 those annotations can be evaluated at import time and fail before
users reach KeepGPU's platform detection, controllers, or tests.

## Goal

Keep modules with PEP 604 annotations import-compatible with the configured
Python 3.9 target by requiring postponed annotations where that syntax appears.

## Solution

- Add a static regression test that finds Python files containing PEP 604 union
  annotations without `from __future__ import annotations`.
- Add the future import to the small set of affected source/test modules.
- Update `AGENTS.md` so future edits keep Python 3.9-targeted annotations safe.

## Tasks

- [x] Add RED compatibility test.
- [x] Add postponed annotations to affected files.
- [x] Update `AGENTS.md`.
- [x] Run targeted compatibility tests.
- [x] Run broader verification.
- [ ] Commit, push, PR, local review, hosted review, and squash merge only after
  all comments/checks are resolved.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_annotation_compat.py -q` failed with
  `src/keep_gpu/utilities/platform_manager.py`,
  `tests/rocm_controller/test_rocm_utilization.py`, and
  `tests/utilities/test_gpu_monitor.py` as offenders.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_annotation_compat.py -q` passed.
- Targeted:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_platform_manager.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py -q`
  passed with 41 passed.
- Compile:
  `python -m py_compile src/keep_gpu/utilities/platform_manager.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py`
  passed.
- Reviewer RED:
  after broadening the guard to AST-detect any PEP 604 union in annotations,
  `PYTHONPATH=$PWD/src pytest tests/test_annotation_compat.py -q` failed with
  `src/keep_gpu/single_gpu_controller/cuda_gpu_controller.py`,
  `src/keep_gpu/single_gpu_controller/macm_gpu_controller.py`,
  `src/keep_gpu/single_gpu_controller/rocm_gpu_controller.py`, and
  `src/keep_gpu/utilities/logger.py` as additional offenders.
- Reviewer GREEN:
  after adding future annotations to those additional offenders,
  `PYTHONPATH=$PWD/src pytest tests/test_annotation_compat.py -q` passed.
- Reviewer targeted:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_platform_manager.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py tests/cuda_controller tests/rocm_controller tests/macm_controller -q`
  passed with 68 passed, 9 skipped.
- Reviewer compile:
  `python -m py_compile src/keep_gpu/utilities/platform_manager.py src/keep_gpu/utilities/logger.py src/keep_gpu/single_gpu_controller/cuda_gpu_controller.py src/keep_gpu/single_gpu_controller/rocm_gpu_controller.py src/keep_gpu/single_gpu_controller/macm_gpu_controller.py tests/utilities/test_gpu_monitor.py tests/rocm_controller/test_rocm_utilization.py tests/test_annotation_compat.py`
  passed.
- Local review:
  the reviewer found `_has_future_annotations()` could falsely flag a valid
  file when another `__future__` import preceded `annotations`; a RED helper
  regression failed, then passed after scanning the full top-of-file future
  import block.
- Hosted review:
  accepted Gemini's `_annotation_nodes()` simplification using `ast.arg`;
  rejected the whole-body `__future__` scan because `ast.parse()` accepts
  misplaced future imports, then added a regression that keeps misplaced
  annotations imports from being treated as valid. Accepted CodeRabbit's tracked
  repository-wide scan and `Annotated` metadata false-positive findings with
  regressions for both.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 583 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan-page notices.
- Hygiene:
  `pre-commit run --all-files` and `git diff --check` passed.
