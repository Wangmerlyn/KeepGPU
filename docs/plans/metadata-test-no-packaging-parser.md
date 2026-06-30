# Metadata Test Parser Cleanup Plan

## Background

`tests/test_package_metadata.py` imports `packaging` only to extract dependency
names while checking package metadata. The project does not declare `packaging`
as a direct runtime or dev dependency, so the test currently relies on pytest or
the environment to provide it transitively.

## Goal

Keep metadata tests self-contained without adding dependency surface.

## Solution

- Add a small parser behavior test for dependency names with versions, extras,
  markers, and mixed punctuation.
- Replace the `packaging` imports with a tiny local dependency-name normalizer.
- Document the guardrail that metadata tests should avoid undeclared parser
  dependencies for simple checks.

## Todo

- [x] Add RED dependency-name parser test.
- [x] Remove the test-only `packaging` import.
- [x] Update agent and contributor guidance.
- [x] Run targeted metadata tests, pre-commit, docs build, and full tests.
- [x] Request local subagent code review before opening the PR.
