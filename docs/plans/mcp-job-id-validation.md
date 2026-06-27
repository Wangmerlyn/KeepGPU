# MCP Job ID Validation Plan

## Background

`KeepGPUServer.stop_keep(job_id="")` currently follows the same branch as
`job_id=None`, so a malformed targeted stop can release every active session.
Custom `job_id` values are also accepted as any Python object. That can create
integer-keyed sessions that REST paths cannot address, or path-shaped IDs that
make `/api/sessions/{job_id}` ambiguous.

## Goal

Make custom session identifiers explicit, stable, and safe across Python,
JSON-RPC, REST, CLI, and dashboard-facing service contracts. Only `None` means
"omitted"; invalid custom IDs must fail before a session starts or stops.

## Design

- Add a shared `validate_job_id()` helper in
  `src/keep_gpu/utilities/session_config.py`.
- Keep `None` as the only omitted/all-sessions sentinel.
- Require custom `job_id` values to be non-empty URL-path-safe strings without
  trimming or other normalization, so a REST session URL maps to exactly one
  session key.
- Use the helper in `KeepGPUServer.start_keep()`, `stop_keep()`, and `status()`.
- Let JSON-RPC and REST reuse the same server validation instead of carrying a
  parallel rule set.
- Update README, MCP/API docs, and `AGENTS.md` so contributors keep the session
  contract aligned.

## Todo

- [x] Add failing unit tests for `validate_job_id()` accepting `None` and
      safe strings while rejecting empty, non-string, whitespace, slash, and
      query-shaped IDs.
- [x] Add failing server tests proving `stop_keep(job_id="")` and JSON-RPC
      `stop_keep` with an empty ID return errors without stopping active
      sessions.
- [x] Add failing service tests proving invalid custom IDs are rejected at
      session start and do not create sessions.
- [x] Add failing REST path tests proving extra or decoded-invalid job ID path
      segments are rejected without stopping another session.
- [x] Address local review feedback with failing tests for targeted REST query
      strings and CLI explicit empty `--job-id` forwarding.
- [x] Address re-review feedback with failing tests for REST semicolon path
      parameters before targeted session lookup or stop.
- [x] Address final local review feedback with failing tests for malformed REST
      collection paths before stop-all can run.
- [x] Implement shared `job_id` validation and wire it through server start,
      stop, and status paths.
- [x] Update `AGENTS.md`, README, MCP guide, and API reference with the custom
      ID contract.
- [x] Run targeted session/service tests, full tests, docs build, pre-commit,
      and local subagent review before opening the PR.
