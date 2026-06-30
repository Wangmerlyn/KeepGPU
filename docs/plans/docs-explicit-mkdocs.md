# Explicit MkDocs Dependency Plan

## Background

The documentation workflow installs `docs/requirements.txt` and then invokes
`mkdocs build --strict`. The requirements file currently relies on
`mkdocs-material` to provide the `mkdocs` CLI transitively.

## Goal

Keep documentation builds self-contained and explicit by declaring the tools the
docs workflow invokes directly.

## Solution

- Add a workflow test that checks `docs/requirements.txt` includes `mkdocs`.
- Add `mkdocs` as a direct documentation dependency.
- Document the guardrail in agent and contributor guidance.

## Todo

- [x] Add RED docs requirements guard.
- [x] Add the direct MkDocs dependency.
- [x] Update agent and contributor guidance.
- [x] Run targeted workflow tests, docs build, pre-commit, and full tests.
- [x] Request local subagent code review before opening the PR.
