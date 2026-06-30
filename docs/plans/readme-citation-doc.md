# README Citation Cleanup Plan

## Background

`README.md` is the repository front door, but its citation section still embeds
the full BibTeX entry. That makes the front page longer even though detailed
citation metadata belongs naturally in the documentation site.

## Goal

Keep README concise while preserving the full citation information in a
published docs page.

## Solution

- Move the BibTeX entry to `docs/citation.md`.
- Link to the citation page from README and the MkDocs navigation.
- Add a small README-shape guard so the full BibTeX block does not drift back
  into README.
- Document the rule in agent and contributor guidance.

## Todo

- [x] Add RED README citation guard.
- [x] Move the BibTeX entry into docs.
- [x] Update README, MkDocs nav, AGENTS.md, and contributor guidance.
- [x] Run targeted tests, docs build, and pre-commit.
- [ ] Request local subagent review before opening the PR.
