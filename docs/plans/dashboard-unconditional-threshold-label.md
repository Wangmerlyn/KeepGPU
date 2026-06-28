# Dashboard Unconditional Threshold Label Plan

## Background

Dashboard session cards currently render `busy_threshold` by appending `%` to
the raw value. That makes `busy_threshold=-1` appear as `threshold -1%`, even
though `-1` means unconditional keepalive mode with utilization backoff disabled.

## Goal/Solution

Render threshold labels semantically in the dashboard. Add a small formatter in
`web/dashboard/src/lib/session.js` that returns `unconditional` for `-1` and
normal percentage labels for thresholds from `0` through `100`, then use that
formatter in `web/dashboard/src/App.jsx` session cards.

## Tasks

- [x] Add RED Vitest coverage for `formatBusyThresholdLabel(-1)`,
      `formatBusyThresholdLabel(0)`, `formatBusyThresholdLabel(25)`, and
      `formatBusyThresholdLabel(100)`.
- [x] Implement the formatter and wire session-card threshold rendering through
      it.
- [x] Update `AGENTS.md` with the dashboard sentinel-label guideline.
- [x] Rebuild tracked dashboard static assets under `src/keep_gpu/mcp/static/`.
- [x] Run focused and requested verification commands.

## Verification Notes/Results

- RED focused dashboard test:
  `npm --prefix web/dashboard test -- src/lib/session.test.js` failed because
  `formatBusyThresholdLabel` was not yet a function.
- GREEN focused dashboard test:
  `npm --prefix web/dashboard test -- src/lib/session.test.js` passed after the
  formatter and App usage were added.
- Full dashboard suite:
  `npm --prefix web/dashboard test` passed with 3 files and 42 tests.
- Dashboard static build:
  `npm --prefix web/dashboard run build` passed and emitted
  `src/keep_gpu/mcp/static/index.html`,
  `src/keep_gpu/mcp/static/assets/index.css`, and
  `src/keep_gpu/mcp/static/assets/dashboard.js`.
- Focused MCP regression:
  `PYTHONPATH=$PWD/src pytest tests/mcp/test_http_api.py::test_http_session_start_preserves_explicit_unconditional_busy_threshold -q`
  passed with 1 test.
