# Fractional Interval Parity Plan

KeepGPU's core Python validation already accepts finite positive fractional
seconds, and the Python API documents examples such as `interval=0.5`. Some
public surfaces still narrow that contract to integers:

- Typer parses CLI `--interval` options as `int`, so `0.5` fails before shared
  validation runs.
- The MCP `start_keep` tool schema advertises `interval` as an integer.
- The dashboard form and payload helper reject fractional interval values.
- Reference docs still show `--interval INTEGER`.

Goal: make every first-class public surface accept and document finite positive
seconds, including fractional seconds, while keeping `busy_threshold` and GPU
IDs integer-only.

Approach:

- Add RED tests for CLI blocking/start commands, MCP schema, direct
  server/JSON-RPC/REST preservation, and dashboard payload/form behavior.
- Keep `validate_interval()` as the single runtime source of truth.
- Parse CLI interval text locally so fractional input reaches
  `validate_interval()` and integer-looking input remains an integer.
- Change the MCP schema to `number` with `exclusiveMinimum: 0`.
- Replace dashboard interval integer parsing with finite-positive-number
  parsing and make the HTML input fractional-friendly.
- Update `AGENTS.md`, README, CLI/MCP/Python/API docs, and generated dashboard
  assets.

Todo:

- [x] Add RED tests for fractional interval parity.
- [x] Implement CLI, MCP schema, dashboard, and type/doc fixes.
- [x] Rebuild packaged dashboard assets.
- [x] Run targeted and full verification.
- [x] Run local subagent review and resolve findings.
- [x] Open and squash-merge implementation PR #116, resolve hosted
      comments/checks, and clean the fractional interval worktree.
