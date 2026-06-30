# Docs MkDocs Major Bound Plan

## Background

`mkdocs build --strict` passes, but Material for MkDocs emits a warning that
MkDocs 2.0 will break plugins and theme overrides. CI installs
`docs/requirements.txt` directly, and the file asked for unbounded `mkdocs`.

## Goal

Keep docs CI stable by bounding the known incompatible MkDocs major while leaving
the docs toolchain simple and direct.

## Tasks

- [x] Create an isolated worktree branch from latest `main`.
- [x] Reproduce the current Material/MkDocs 2.0 warning.
- [x] Add a RED guard requiring the known incompatible MkDocs major bound.
- [x] Add `mkdocs<2` and update `AGENTS.md`.
- [x] Run docs/tests.
- [ ] Run local subagent review.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_ci_workflows.py::test_docs_requirements_bound_known_incompatible_mkdocs_major -q`
  failed because the active requirement was `mkdocs`.
- GREEN:
  the same command passed after changing the requirement to `mkdocs<2`.
- Broader checks:
  `PYTHONPATH=$PWD/src pytest tests/test_ci_workflows.py -q` passed with
  `8 passed` after the guard was hardened against `mkdocs<20` and `mkdocs<2.1`;
  `PYTHONPATH=$PWD/src pytest -q` passed with `782 passed, 11 skipped`;
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed while still showing the
  existing Material proactive warning.
