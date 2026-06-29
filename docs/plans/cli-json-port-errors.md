# CLI JSON Port Errors Plan

## Background

`keep-gpu status`, `keep-gpu stop --all`, and `keep-gpu list-gpus` are
JSON-output commands. Invalid endpoint values should therefore return a single
machine-readable `{"error": "..."}` object before RPC or stop-all fallback
logic runs. Non-integer `--port` tokens were rejected by Typer/Click before the
command body, producing usage text and exit code 2 instead of the established
JSON error contract.

## Goal

Keep JSON-output service commands directly parseable even when users or agents
pass malformed port values.

## Solution

- Add command-level regression coverage for non-integer `--port` values on
  `status`, `stop --all`, and `list-gpus`.
- Let JSON-output commands receive the raw port token and route it through the
  shared endpoint validator.
- Extend the shared service port validator to parse string tokens and return the
  normalized integer port after the existing range check.
- Document that non-integer and out-of-range ports on JSON-output commands are
  reported as structured JSON errors before service side effects.

## Tasks

- [x] Reproduce the current plain-text Click usage error for
      `keep-gpu status --port abc`.
- [x] Add RED tests for non-integer `--port` on JSON-output service commands.
- [x] Implement shared string port parsing for JSON-output command paths.
- [x] Update `AGENTS.md`, README, CLI guide/reference, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and
      `git diff --check`.
- [x] Run local subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge,
      and clean the worktree.

## Verification

- Symptom reproduction:
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli status --port abc` returned Click
  usage text with exit code 2 before the fix.
- RED regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_service_json_commands_reject_non_integer_port_as_json_before_rpc_or_fallback -q`
  failed with three exit-code assertions (`2 == 1`), showing that Click rejected
  each command before JSON rendering.
- GREEN focused regression:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_service_json_commands_reject_non_integer_port_as_json_before_rpc_or_fallback -q`
  passed with `3 passed`.
- GREEN CLI service shard:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q` passed with
  `148 passed`.
- Live symptom check:
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli status --port abc`,
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli stop --all --port abc`, and
  `PYTHONPATH=$PWD/src python -m keep_gpu.cli list-gpus --port abc` each
  returned a single `{"error": "port must be an integer between 1 and 65535"}`
  JSON object with exit code 1.
- Focused CLI gate:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q`
  passed with `162 passed`.
- Full no-GPU-safe gate:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with `597 passed, 11 skipped`.
- Docs and hygiene:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the repository's existing
  Material warning and unnav'd plan notices; `pre-commit run --all-files`
  passed; and `git diff --check` passed.
- Local subagent review:
  The reviewer found no critical, important, or minor issues, reran the live
  non-integer port checks, reran the focused regression, and marked the branch
  ready to merge.
