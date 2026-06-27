# Eco-Safe Default Busy Threshold Plan

## Background

KeepGPU now treats unknown utilization telemetry conservatively when
`busy_threshold` is non-negative, but several public entry points still default
to `busy_threshold=-1`. That makes omitted options opt into unconditional
keepalive compute, even though the dashboard and the product direction are
power-aware and polite by default.

## Goal

Make omitted public session defaults eco-safe by using a non-negative
`busy_threshold` of `25`, while preserving explicit `busy_threshold=-1` as the
unconditional keepalive opt-in.

## Solution

- Add a shared `DEFAULT_BUSY_THRESHOLD = 25` constant beside public session
  validation in `src/keep_gpu/utilities/session_config.py`.
- Route CLI blocking mode, CLI service mode, MCP direct calls, MCP tool schema,
  and REST session creation through the shared default.
- Keep validation unchanged: `-1` and `0..100` remain valid, and explicit `-1`
  keeps its current meaning.
- Update docs and `AGENTS.md` so contributors preserve eco-safe omitted
  defaults and document `-1` as an explicit high-power mode.

## Tasks

- [x] Add failing tests for default `busy_threshold=25` in:
  - blocking CLI invocation without `--busy-threshold`,
  - service CLI `start` without `--busy-threshold`,
  - `KeepGPUServer.start_keep()` without `busy_threshold`,
  - JSON-RPC/HTTP session creation without `busy_threshold`,
  - MCP `tools/list` schema default.
- [x] Implement the shared default constant and use it in CLI and MCP/server
      defaults.
- [x] Preserve explicit `--busy-threshold -1` and explicit JSON payload
      `busy_threshold=-1` behavior.
- [x] Update `AGENTS.md`, README, CLI reference/guide, MCP guide, and Python/API
      guidance where defaults or default semantics are described.
- [x] Run targeted tests, full tests, docs build, and pre-commit.
- [x] Request local subagent review before opening the PR.
- [ ] Open PR, resolve all review comments, wait for clean GitHub checks, then
      squash merge and clean the worktree.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/test_cli_thresholds.py tests/test_cli_service_commands.py tests/mcp tests/global_controller/test_contract.py -q`: 135 passed.
- `PYTHONPATH=$PWD/src pytest tests -q`: 224 passed, 12 skipped.
- `PYTHONPATH=$PWD/src mkdocs build`: passed with existing Material/MkDocs warning and unnav'd plan-page notices.
- `pre-commit run --all-files`: passed.
- Local subagent spec review: no must-fix issues; reviewer also ran targeted tests (135 passed) and `git diff --check origin/main`.
- Local subagent code-quality review: no must-fix issues; reviewer confirmed shared default use and explicit `-1` behavior.
