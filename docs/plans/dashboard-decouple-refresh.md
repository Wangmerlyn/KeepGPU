# Dashboard Refresh Decoupling Plan

## Background

The dashboard refreshed `/api/gpus` and `/api/sessions` with one `Promise.all`.
If telemetry failed while session status succeeded, session state stayed stale.
After start or release actions, a follow-up telemetry warning could also replace
the successful action result message.

## Goal

Keep the dashboard useful when telemetry is flaky: apply whichever refresh
payload succeeds, warn on partial failure, and preserve mutation outcome
messages after start/release operations.

## Solution

- Add RED refresh-helper tests for telemetry failure with successful sessions
  and mutation-message preservation.
- Fetch dashboard payloads with independent settled outcomes.
- Update `App.jsx` to apply successful payloads independently and keep mutation
  messages as the primary footer text.
- Rebuild the packaged static dashboard assets.

## Verification

- Dashboard refresh helper tests.
- Full dashboard test/build.
- Relevant Python/API checks, pre-commit, docs build, and whitespace check
  before PR.
