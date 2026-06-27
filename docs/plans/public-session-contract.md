# Public Session Contract Fix Plan

## Background

Audit agents found four related public-contract bugs: unitless `--vram` strings are documented as bytes but parse as GB, CLI/MCP defaults can suppress keep-alive compute, zero or negative intervals can create tight loops, and CLI/JSON-RPC accept negative GPU IDs while REST rejects them.

## Goal

Make CLI, Python service, REST, and JSON-RPC use the same humane validation and default policy for session inputs while keeping the change focused.

## Solution

- Keep human-readable VRAM strings such as `512MB` and `1GiB`.
- Treat digit-only VRAM strings and public integer `vram_to_keep` values as bytes.
- Convert public bytes to internal float32 element counts once, close to controller setup.
- Treat negative `busy_threshold` as "utilization backoff disabled" instead of "always back off".
- Reject non-positive intervals and negative GPU IDs consistently across CLI, direct service calls, REST, and JSON-RPC.
- Update docs and AGENTS guidance for the clarified public contract.

## Todo

- [x] Add failing tests for VRAM byte parsing, interval validation, GPU ID validation, and busy-threshold semantics.
- [x] Implement shared validation/parsing helpers without scattering platform branches.
- [x] Update CLI/MCP/Python controller paths to use the shared contract.
- [x] Update README, docs, and AGENTS.md with the clarified contract and validation expectation.
- [x] Run targeted tests, docs build, and pre-commit.
- [ ] Open PR, run local subagent review, resolve all comments, then squash merge only when clean.
