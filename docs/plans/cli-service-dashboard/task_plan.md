# Task Plan: Non-blocking CLI + Local Service + Dashboard

## Background

`keep-gpu` currently runs as a blocking command. This works for manual shells but is awkward for LLM agent workflows that need to continue with follow-up commands.

## Goal

Add an agent-friendly, non-blocking CLI workflow while preserving the existing blocking behavior, then provide a production-grade web dashboard backed by the same local service.

## Proposed Solution

1. Keep existing `keep-gpu` (blocking) unchanged for compatibility.
2. Extend CLI with service-driven subcommands: `serve`, `start`, `status`, `stop`, and `list-gpus`.
3. Implement service auto-start in `start` when local service is unavailable.
4. Extend HTTP server with REST endpoints and serve a React/Vite-built dashboard.
5. Update docs and the KeepGPU skill to reflect the new preferred agent workflow.

## Phases

### Phase 1: Design and Scaffolding
- [x] Define CLI subcommand UX and local service contract
- [x] Add service client/autostart helpers
- [x] Add REST handlers and static file serving skeleton
- **Status:** complete

### Phase 2: Dashboard Frontend (React/Vite)
- [x] Add Vite app and distinctive UI design
- [x] Implement live GPU/session views and session controls
- [x] Build and bundle static assets for Python serving
- **Status:** complete

### Phase 3: Tests and Documentation
- [x] Add/extend tests for new CLI and server behavior
- [x] Update CLI/MCP docs and README
- [x] Update `skills/gpu-keepalive-with-keepgpu/SKILL.md`
- **Status:** complete

### Phase 4: Validation
- [x] Run targeted pytest suites
- [x] Run `pre-commit run --all-files`
- [x] Run `mkdocs build`
- **Status:** complete

## Risks / Open Questions

1. How to keep auto-start robust without introducing zombie processes.
2. How to package dashboard assets cleanly for source and installed usage.
