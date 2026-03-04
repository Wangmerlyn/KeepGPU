# Findings: Non-blocking CLI + Dashboard

## Repository Observations

- Blocking CLI entrypoint is `src/keep_gpu/cli.py` (`main` Typer command).
- Existing remote/session model already exists in `src/keep_gpu/mcp/server.py` (`start_keep`, `stop_keep`, `status`, `list_gpus`).
- Existing MCP HTTP mode only accepts JSON-RPC `POST` and does not expose REST or web UI.
- Current tests for MCP are in `tests/mcp/test_server.py` and cover core session methods.

## Design Direction (Frontend Skill)

- Aesthetic direction: **retro-futuristic control room**.
- Visual memory hook: glowing telemetry grid + layered glass panels + high-contrast amber/cyan instrumentation.
- UX priority: one-screen control loop (start session, inspect health, stop session) for agent operators.

## Implementation Notes

- Keep backward compatibility: default `keep-gpu` remains blocking.
- Non-blocking usage should be first-class through subcommands and local auto-start.
- Dashboard should be served by the same local backend to avoid extra runtime coordination.

## Implemented Architecture

- `src/keep_gpu/cli.py` now exposes:
  - blocking default callback (backward compatible),
  - `serve`, `start`, `status`, `stop`, and `list-gpus` subcommands,
  - local service auto-start in `start` by launching `python -m keep_gpu.mcp.server`.
- `src/keep_gpu/mcp/server.py` now exposes REST routes and static UI serving:
  - `GET /health`, `GET /api/gpus`, `GET /api/sessions`, `GET /api/sessions/{job_id}`
  - `POST /api/sessions`, `DELETE /api/sessions`, `DELETE /api/sessions/{job_id}`
  - dashboard assets on `/` with SPA fallback.
- Dashboard source lives in `web/dashboard/` (React + Vite), with build output in
  `src/keep_gpu/mcp/static/`.

## Validation Findings

- New CLI and server tests pass (`13 passed`).
- Pre-commit hooks pass with no auto-fix needed after initial implementation.
- `mkdocs build` succeeds; informational note reports plan/skills docs excluded from nav.

## Iteration Findings (Post-UX Feedback)

- Added explicit daemon lifecycle guidance in CLI output and docs.
- Added `keep-gpu service-stop` command so users can terminate auto-started local daemon.
- Updated `keep-gpu start` output to always print dashboard URL and follow-up stop commands.
- Reworked dashboard visual language from neon/glassy to classic, restrained control-room style.
- Expanded CLI tests to cover start-output hints and `service-stop` guardrails.

## Iteration Findings (Bug Report Follow-up)

- Root cause for noisy `keep-gpu stop --all` failure output:
  - service commands still imported `torch` at module import time, causing unrelated NumPy warning noise,
  - network timeout from `urlopen` was not normalized, so users could see raw traceback-style failures.
- Fixes applied:
  - moved `torch` import into blocking-only execution path,
  - wrapped HTTP/timeout errors in friendly `RuntimeError` messages,
  - extended command error handling to avoid traceback leaks to users,
  - improved dashboard action state so releasing one session does not disable every release control globally.
- Additional tests now assert timeout/error behavior for `keep-gpu stop --all` and HTTP wrapper timeout handling.

## Root-Cause Fix Findings (Stop-All Reliability)

- Reproduced that frequent stop timeouts are mostly service-side latency under release, not network transport.
- Root causes identified:
  - worker loops used `time.sleep(interval)`, which is non-interruptible and can delay shutdown for long intervals,
  - stop RPC could block behind slow release execution,
  - single-threaded HTTP server amplified perceived hangs under long requests.
- Fixes implemented:
  - replaced sleep points in CUDA/ROCm keep loops with `Event.wait(interval)` so stop signals interrupt immediately,
  - added bounded release join timeout in CUDA/ROCm controllers to avoid indefinite blocking,
  - made release-all server responses timeout-aware (`timed_out` reporting) with background continuation,
  - switched HTTP server to threaded mode to prevent one slow request from blocking others.
