# CI Root Requirements Cleanup Plan

## Background

KeepGPU uses `pyproject.toml` for package dependencies and `docs/requirements.txt`
for documentation builds. The Python application workflow still has a legacy
fallback that installs root `requirements.txt` if someone adds that file later,
which can silently widen the test dependency surface.

## Goal

Keep Python CI dependency installation explicit and lean by removing the root
`requirements.txt` fallback and adding a guard against reintroducing it.

## Solution

- Add a CI workflow test that fails if the Python application workflow installs
  root `requirements.txt`.
- Remove the conditional `pip install -r requirements.txt` line.
- Document the guardrail in `AGENTS.md` and `docs/contributing.md`.

## Todo

- [x] Add RED workflow guard.
- [x] Remove the legacy root requirements install fallback.
- [x] Update agent and contributor guidance.
- [x] Run targeted workflow tests and pre-commit.
- [x] Request local subagent code review before opening the PR.
