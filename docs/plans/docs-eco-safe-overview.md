# Docs Eco-Safe Overview Plan

## Background

The public overview claimed that KeepGPU continuously allocates VRAM and makes a
GPU always look in use. That overstates the default behavior because unavailable
telemetry makes the default loop back off when `busy_threshold >= 0`.

## Goal

Keep the overview concise while making the eco-safe contract clear: allocation
and keepalive work happen only when utilization backoff permits, and
`busy_threshold=-1` is the explicit unconditional mode.

## Tasks

- [x] Add a RED doc guard for unconditional overview claims.
- [x] Update `docs/index.md` with the backoff caveat.
- [x] Add an `AGENTS.md` contract note for future overview edits.
- [x] Run the focused guard.
- [x] Run the strict docs build.

## Verification

- RED:
  `pytest tests/test_package_metadata.py::test_index_overview_describes_eco_safe_backoff_without_unconditional_claims -q`
  failed because `docs/index.md` still said `continuously allocates`.
- GREEN:
  the same focused guard passed after the overview described unavailable
  telemetry backoff and the explicit unconditional mode.
- DOCS:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the existing Material
  for MkDocs warning.
