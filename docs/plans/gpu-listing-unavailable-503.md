# GPU Listing Unavailable REST Plan

## Background

`POST /api/sessions` already treats expected GPU enumeration failures as
service-unavailable startup conditions, but `GET /api/gpus` let the same
`DeviceEnumerationUnavailableError` fall through the generic REST runtime-error
handler as HTTP 500.

## Goal

Keep `/api/gpus` structured and actionable: expected enumeration unavailability
returns HTTP 503, while arbitrary listing failures remain structured HTTP 500.

## Solution

- Add a RED HTTP regression for `GET /api/gpus` raising
  `DeviceEnumerationUnavailableError`.
- Catch that expected exception in the `/api/gpus` route before the generic
  runtime-error handler.
- Update the REST/MCP docs and agent guidance for the new classification.

## Verification

- Focused `/api/gpus` unavailable and runtime-error tests.
- MCP HTTP API shard.
- Full tests, pre-commit, docs build, and whitespace check before PR.
