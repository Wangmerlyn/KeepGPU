# CLI Service Endpoint Validation Plan

## Background

Service-mode CLI commands build URLs directly from `--host` and `--port`.
Malformed hosts such as `bad host` can leak lower-level URL tracebacks from
JSON-output commands, and `keep-gpu start` can reach daemon auto-start logic
before rejecting the endpoint. That violates the CLI contract that local bad
inputs fail before service side effects and that JSON commands print parseable
error objects.

## Goal

Reject invalid service endpoint flags locally before service RPC, daemon
auto-start, stop-all fallback, or daemon ownership operations.

## Solution

- Add shared CLI service endpoint validation for `--host` and `--port`.
- Apply it to `serve`, `start`, `status`, `stop`, `list-gpus`, and
  `service-stop`.
- Keep JSON-output commands (`status`, `stop`, `list-gpus`) returning a single
  `{"error": "..."}` object for invalid endpoints.
- Wrap lower-level malformed URL `InvalidURL`/`ValueError` failures as
  `ServiceUnreachableError` as a defensive fallback.
- Update docs and `AGENTS.md` with the no-side-effect endpoint contract.

## Tasks

- [x] Add RED CLI regression tests for invalid `--host` before RPC,
      auto-start, or stop-all fallback.
- [x] Implement shared service endpoint validation and command integration.
- [x] Add explicit invalid-port regression coverage.
- [x] Update `AGENTS.md`, README, CLI guide/reference, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and `git diff
      --check`.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'invalid_host'`
  failed with four cases because `start`, `status`, `stop --all`, and
  `list-gpus` continued into service/RPC paths with `--host "bad host"`.
- GREEN focused endpoint regressions:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'invalid_host or invalid_port'`
  passed with `6 passed, 68 deselected`.
- GREEN CLI service-command shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed
  with `76 passed` after adding defensive malformed-URL and `service-stop`
  endpoint coverage.
- Live symptom check:
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli status --host 'bad host'` and
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli list-gpus --host 'bad host'`
  returned parseable JSON error objects instead of tracebacks.
- Broad verification before local review:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with `87 passed`; `PYTHONPATH=$PWD/src pytest tests -q` passed with
  `415 passed, 11 skipped`; `PYTHONPATH=$PWD/src mkdocs build` passed with the
  repository's existing Material warning and unnav'd plan notices; and
  `pre-commit run --all-files` plus `git diff --check` passed.
- Local review follow-up: reviewer found that the first host guard still
  accepted malformed non-whitespace values such as `%` and `\host`. Tightened
  validation to DNS hostnames or IPv4 addresses, added syntax-level host tests,
  added a non-whitespace malformed-host command regression, documented the
  `service-stop` endpoint safety contract, and reran
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'validate_cli_service_host or non_whitespace_malformed_host or invalid_host or invalid_port or service_stop'`
  with `32 passed, 66 deselected` and
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` with
  `98 passed`.
- Final branch gate after local-review follow-up:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with `109 passed`; `PYTHONPATH=$PWD/src pytest tests -q` passed with
  `437 passed, 11 skipped`; `PYTHONPATH=$PWD/src mkdocs build` passed with the
  repository's existing Material warning and unnav'd plan notices; and
  `pre-commit run --all-files` plus `git diff --check` passed.
- Second local subagent review: passed with no critical, important, or minor
  findings. Reviewer reran the endpoint shard with `32 passed, 66 deselected`,
  confirmed malformed host/port live checks return single JSON error objects,
  and confirmed `git diff --check`.
- Gemini review follow-up: rejected numeric single-label hosts and numeric TLDs
  such as `123` and `foo.123`, and changed invalid UTF-8 service bodies from a
  misleading unreachable-service error to `ServiceResponseError`. The focused
  review-fix shard passed with `35 passed, 66 deselected`. Final post-review
  gates passed: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  with `112 passed`; `PYTHONPATH=$PWD/src pytest tests -q` with `440 passed,
  11 skipped`; `PYTHONPATH=$PWD/src mkdocs build`; `pre-commit run
  --all-files`; and `git diff --check`.
