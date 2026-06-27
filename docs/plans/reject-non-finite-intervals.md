# Reject Non-Finite Intervals Plan

## Background

KeepGPU treats `interval` as the sleep between telemetry checks and keepalive
bursts. The shared validator currently accepts any numeric value greater than
zero. That lets `float("nan")` through because comparisons with NaN are false,
and JSON request bodies can contain `NaN`/`Infinity` when parsed by Python's
default `json.loads`.

## Goal

Reject `NaN`, positive infinity, and negative infinity anywhere public interval
validation is used so malformed input cannot create tight or crashing keep
loops.

## Solution

- Add failing tests for non-finite values in shared interval validation.
- Add failing JSON-RPC and REST tests proving `interval: NaN` does not create a
  session.
- Use `math.isfinite()` in `validate_interval()` after confirming the value is a
  plain non-boolean number.
- Update AGENTS and user docs to state that intervals must be finite positive
  seconds.

## Tasks

- [x] Add RED tests for `validate_interval(math.nan)`,
      `validate_interval(math.inf)`, and `validate_interval(-math.inf)`.
- [x] Add RED JSON-RPC test for `interval: NaN` rejecting startup without
      creating a session.
- [x] Add RED REST test for `interval: NaN` returning HTTP 400 without creating
      a session.
- [x] Implement finite positive interval validation in
      `src/keep_gpu/utilities/session_config.py`.
- [x] Update `AGENTS.md`, README, CLI/Python/MCP/API docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  failed with 5 expected failures covering `NaN`/infinity validation and MCP
  session creation.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/mcp/test_server.py tests/mcp/test_http_api.py -q`
  passed with 127 tests.
- `PYTHONPATH=$PWD/src pytest tests -q` passed with 240 tests and 11 skipped.
- `PYTHONPATH=$PWD/src mkdocs build` passed with existing Material/MkDocs and
  unlisted-plan warnings.
- `pre-commit run --all-files` passed.
- `git diff --check` passed.
- Local subagent review found one packaging issue: this plan file was untracked.
  The fix is to include the plan file in the branch before opening the PR.
