# Dashboard Unknown Utilization Summary Plan

The dashboard currently computes average utilization with
`gpu.utilization ?? 0`, so unsupported or unavailable telemetry can appear as
`0%` idle. Per-GPU cards also render unknown utilization as `n/a%`. That conflicts
with KeepGPU's eco-safe contract: unavailable telemetry is not idle telemetry.

Goal: make the dashboard display unavailable utilization as unknown/`n/a` and
average only known numeric utilization values.

Approach:

- Move dashboard stats and utilization-label formatting into the existing
  `web/dashboard/src/lib/session.js` helper module.
- Keep the UI simple: the summary remains `n/a` when no utilization readings are
  known; mixed telemetry averages only finite numeric readings.
- Update dashboard docs and `AGENTS.md` so future UI changes do not convert
  unknown telemetry into an idle signal.
- Rebuild the tracked static dashboard bundle under `src/keep_gpu/mcp/static`.

Todo:

- [x] Add RED Vitest helper tests for all-unknown, mixed-known, and per-card
      utilization labels.
- [x] Implement the stats/label helpers and wire them into `App.jsx`.
- [x] Update `AGENTS.md`, README, and MCP/dashboard docs.
- [x] Run dashboard tests, dashboard build, focused checks, full tests, docs
      build, pre-commit, and `git diff --check`.
- [ ] Run local subagent review, resolve findings, open PR, resolve hosted
      comments/checks, squash merge, and clean the worktree.
