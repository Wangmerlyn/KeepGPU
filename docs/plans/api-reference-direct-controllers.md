# API Reference Direct Controllers Plan

## Background

The API reference explains the public behavior of CUDA, ROCm, and Mac M
single-GPU controllers, and the Python guide shows direct controller imports.
However, the mkdocstrings section renders only the CUDA direct controller plus
the global controller and utilities. ROCm and Mac M users cannot inspect the
matching direct-controller API from the reference page.

## Goal

Keep the public API reference complete without expanding the README or adding a
new docs page: render the existing ROCm and Mac M controller modules alongside
the CUDA direct controller.

## Solution

- Add mkdocstrings blocks for `rocm_gpu_controller` and `macm_gpu_controller`
  next to the existing CUDA controller block.
- Keep the surrounding prose unchanged because it already covers the behavior
  contract for all three direct controllers.
- Verify the docs build and a small targeted package/docs check.

## Tasks

- [x] Create an isolated `.worktrees/codex/api-reference-direct-controllers`
  branch from latest `origin/main`.
- [x] Add missing ROCm and Mac M direct-controller reference blocks.
- [x] Run docs/full verification and local subagent review.
- [ ] Open a PR, resolve hosted comments/checks, squash merge, and clean up.

## Verification

- Docs:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the existing
  Material for MkDocs warning.
- Targeted:
  `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q` passed with
  11 passed.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 862 passed, 11 skipped.
- Hooks:
  `pre-commit run --all-files --show-diff-on-failure` and `git diff --check`
  passed.
- Local review:
  a local subagent review found no must-fix issues.
