# Progress Log: Non-blocking CLI + Dashboard

## Session 2026-03-04

### Completed

- Created implementation plan in `docs/plans/cli-service-dashboard/task_plan.md`.
- Captured initial repository findings in `docs/plans/cli-service-dashboard/findings.md`.
- Confirmed branch: `feat/cli-service-dashboard`.
- Implemented CLI subcommands and service auto-start in `src/keep_gpu/cli.py`.
- Extended server HTTP endpoints and static dashboard routing in `src/keep_gpu/mcp/server.py`.
- Added React/Vite dashboard source under `web/dashboard/` and built static assets.
- Added tests in `tests/mcp/test_http_api.py` and `tests/test_cli_service_commands.py`.
- Updated README/docs/skill content for new service and dashboard workflow.
- Addressed UX feedback:
  - refined dashboard styling to a classic high-quality aesthetic,
  - improved `keep-gpu start` messaging with dashboard URL and shutdown hints,
  - added `keep-gpu service-stop` command for daemon shutdown.
- Expanded tests for output/help lifecycle behavior.
- Applied follow-up reliability fixes from real-user reproduction:
  - moved `torch` import into blocking-only path,
  - normalized service timeout/HTTP failures into user-friendly errors,
  - fixed dashboard release-button state handling to avoid global lockouts.
- Implemented root-cause shutdown fixes:
  - converted CUDA/ROCm loop sleeps to interruptible waits,
  - added bounded join behavior in controller `release()`,
  - updated server stop behavior to surface timeout-aware payloads,
  - moved HTTP server to threaded mode.

### In Progress

- Preparing final summary and PR after successful validation.

### Next

1. Create commit and open PR for review.

### Validation Results

- `pytest tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_thresholds.py tests/test_cli_service_commands.py` -> 13 passed.
- `pre-commit run --all-files` -> all hooks passed.
- `mkdocs build` -> success (non-blocking info about docs pages outside nav).

### Validation Results (Follow-up)

- `pytest tests/test_cli_service_commands.py tests/mcp/test_http_api.py tests/mcp/test_server.py tests/test_cli_thresholds.py` -> 18 passed.
- `pre-commit run --all-files` -> all hooks passed.
- `mkdocs build` -> success.

### Validation Results (Root-Cause Fix)

- `pytest tests/test_cli_service_commands.py tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_thresholds.py tests/cuda_controller/test_keep_and_release.py tests/cuda_controller/test_throttle.py` -> 26 passed.
- `pre-commit run --all-files` -> all hooks passed.
- `mkdocs build` -> success.

## Error Log (This Iteration)

- `pre-commit run --all-files` reformatted `tests/test_cli_service_commands.py` by way of Black on first run; second run passed.
