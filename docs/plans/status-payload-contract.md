# Status Payload Contract Plan

## Goal

Keep status responses read-only and reject malformed known session params before the CLI renders them as success.

## Scope

- Snapshot server status params for active and starting sessions.
- Validate known nested status params in CLI result handling.
- Keep extra params forward-compatible and avoid broad schema machinery.

## Tasks

- [x] Add failing tests for active and starting status param snapshots.
- [x] Add failing CLI tests for malformed known `status.params` fields.
- [x] Implement a tiny status params snapshot helper in `src/keep_gpu/mcp/server.py`.
- [x] Implement optional known-field status params validation in `src/keep_gpu/cli.py`.
- [x] Update `AGENTS.md` and MCP docs with the status payload contract.
- [ ] Run targeted tests, local code review, PR checks, then squash merge.
