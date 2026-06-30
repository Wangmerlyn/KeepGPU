# Explicit Pymdown Extensions Dependency Plan

## Background

`mkdocs.yml` configures `pymdownx.tabbed` and `pymdownx.emoji` directly, while
`docs/requirements.txt` relies on `mkdocs-material` to provide
`pymdown-extensions` transitively.

## Goal

Keep documentation builds self-contained and explicit by declaring MkDocs
extensions that the project configures directly.

## Solution

- Extend the docs requirements guard to require `pymdown-extensions` when
  `mkdocs.yml` configures `pymdownx.*` extensions.
- Add `pymdown-extensions` as a direct docs dependency.
- Document the direct-extension dependency rule in agent and contributor
  guidance.

## Todo

- [x] Add RED docs extension dependency guard.
- [x] Add the direct `pymdown-extensions` dependency.
- [x] Update agent and contributor guidance.
- [x] Run targeted workflow tests, docs build, pre-commit, and full tests.
- [x] Request local subagent code review before opening the PR.
