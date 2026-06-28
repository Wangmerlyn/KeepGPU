# Dashboard REST Error Messages Plan

The REST service already returns structured JSON errors such as
`{"error":{"message":"Bad request: ..."}}`, but the dashboard currently throws
the raw response body for non-OK requests. That makes the footer show JSON blobs
instead of the human-readable backend message.

Goal: make dashboard failures use the structured REST message first while
preserving plain-text and empty-body fallbacks.

Approach:

- Add a small dashboard API helper that converts failed REST responses into one
  display string and keeps the timeout-aware request wrapper out of `App.jsx`.
- Keep the service/API contract unchanged; this is a frontend parsing fix.
- Wire `App.jsx` to use the helper for all dashboard REST calls.
- Update dashboard/API docs and `AGENTS.md` so future changes keep this UX
  contract aligned.
- Rebuild the tracked static dashboard bundle under `src/keep_gpu/mcp/static`.

Todo:

- [x] Add a RED Vitest helper test proving structured REST JSON displays only
      `error.message`.
- [x] Implement the minimal formatter/request helper and wire it into the
      dashboard.
- [x] Update `AGENTS.md`, README, and MCP/API docs.
- [x] Run dashboard tests, dashboard build, focused MCP tests, full tests, docs
      build, pre-commit, and `git diff --check`.
- [ ] Run local subagent review, resolve findings, open PR, resolve hosted
      comments/checks, squash merge, and clean the worktree.
