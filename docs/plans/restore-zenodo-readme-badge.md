# Restore Zenodo README Badge Plan

## Background

The slim README guard removed badge clutter, but the Zenodo DOI badge is useful
for citation and research-facing credibility.

## Goal

Restore the Zenodo badge without turning README back into a reference page.

## Solution

- Keep README's existing compact front-door copy.
- Add only the Zenodo DOI badge beside PyPI and docs status.
- Update the metadata guard so it requires the exact PyPI, docs status, and DOI
  badge lines.
- Keep detailed citation metadata in `docs/citation.md`.

## Verification

- Targeted package metadata tests.
- README badge URL smoke check.
- Pre-commit and diff whitespace checks before PR.
