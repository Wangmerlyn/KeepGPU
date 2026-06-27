# Busy-Threshold Range Plan

## Background

`busy_threshold` is documented as a utilization percentage, with `-1` reserved
as the explicit unconditional keepalive mode. The current validator rejects
values below `-1`, but accepts values above `100`. A value such as `101` makes a
fully utilized GPU eligible for keepalive compute without using the explicit
`-1` escape hatch.

## Goal

Keep utilization backoff eco-safe by accepting only `-1` or percentages in the
inclusive `0..100` range across the CLI, Python controllers, MCP/JSON-RPC, REST,
and dashboard inputs.

## Design

- Keep `-1` as the only unconditional keepalive sentinel.
- Accept integer thresholds from `0` through `100`, inclusive.
- Reject booleans, non-integers, values below `-1`, and values above `100`
  through the centralized `validate_busy_threshold()` helper.
- Let existing CLI, controller, MCP, REST, and dashboard paths inherit the same
  central rule.
- Update user docs and agent guidance to state the allowed range.

## Todo

- [x] Add failing tests for `101` rejection and `100` acceptance.
- [x] Implement the centralized validator change.
- [x] Add interface-level coverage where cheap to prove CLI/MCP inherit the
      central rule.
- [x] Close local-review parity gap by updating dashboard validation, rebuilding
      static assets, and adding REST rejection coverage.
- [x] Update `AGENTS.md`, README, and CLI/MCP docs with the allowed range.
- [x] Run targeted threshold tests, full tests, docs build, and pre-commit.
- [ ] Open a GitHub PR, run local subagent review, resolve all comments, then
      squash merge.
