# Zenodo Citation Metadata Plan

## Background

The README keeps the Zenodo concept DOI badge as a compact first-viewport
signal. The detailed BibTeX entry lives in `docs/citation.md`, but it drifted
from the current Zenodo metadata for the concept DOI.

## Goal

Keep README compact while making the citation page match the current Zenodo
concept DOI export for KeepGPU v0.5.1.

## Solution

- Add a small metadata guard for the citation page.
- Update `docs/citation.md` with the current Zenodo title, creator list, year,
  DOI, URL, license, and software note.
- Keep README unchanged except for preserving its existing Zenodo badge.

## Checks

- `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q -k 'readme or citation or metadata'`
- `PYTHONPATH=$PWD/src mkdocs build --strict`
- `pre-commit run --all-files --show-diff-on-failure`
