# Fix CLI JSON Output Plan

## Background

`keep-gpu status`, `keep-gpu stop`, and `keep-gpu list-gpus` pass
`json.dumps(result)` as `data` to Rich's `console.print_json()`. Rich treats
that string as data and emits a top-level JSON string, so callers must decode
twice and tools such as `jq` cannot index the output directly.

A later audit found that `console.print_json(data=result)` still routes machine
JSON through Rich rendering. In pseudo-TTY or forced-color environments, Rich can
inject ANSI styling into otherwise structured JSON, breaking a single
`json.loads()` call again.

## Goal

Emit structured JSON objects from service CLI commands so one `json.loads()` or
one shell JSON tool invocation sees the expected object.

## Solution

- Add RED CLI tests for `status`, `stop --job-id`, `stop --all`, and
  `list-gpus` that require a single JSON decode to return a dict.
- Update the existing stop-all fallback test to single-decode the command
  output.
- Print decoded result objects with a small `_print_machine_json()` helper that
  serializes with stdlib `json.dumps()` and writes directly to the console stream
  without Rich color/highlight rendering.
- After CodeRabbit review, print `{"error": "..."}` JSON objects for
  `RuntimeError` paths in the same CLI commands.
- Document in `AGENTS.md`, README, and CLI docs that these CLI commands emit
  directly parseable JSON objects.

## Tasks

- [x] Add RED tests for single-decode structured JSON output.
- [x] Implement minimal CLI output changes.
- [x] Add RED tests and implementation for single-decode JSON error objects.
- [x] Add RED tests and implementation for ANSI-free machine JSON output under
      forced-color consoles.
- [x] Update `AGENTS.md`, README, CLI docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [x] Original PR was opened, reviewed, merged, and the worktree was cleaned.

## Verification

- Baseline: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with 34 tests.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_status_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_job_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_all_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_list_gpus_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_all_fallback_force_stops_managed_daemon -q`
  failed with all five outputs decoding once to strings instead of dicts.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with 38 tests.
- `PYTHONPATH=$PWD/src pytest tests -q` passed with 249 tests and 11 skipped.
- `PYTHONPATH=$PWD/src mkdocs build` passed with existing Material/MkDocs and
  unlisted-plan warnings.
- `pre-commit run --all-files` passed.
- `git diff --check` passed.
- CodeRabbit review:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_stop_requires_job_id_or_all tests/test_cli_service_commands.py::test_status_forwards_explicit_empty_job_id_to_service tests/test_cli_service_commands.py::test_stop_forwards_explicit_empty_job_id_to_service tests/test_cli_service_commands.py::test_list_gpus_error_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_handles_service_timeout_without_traceback -q`
  failed before the error-output fix because command errors still printed Rich
  text, then passed with 5 tests after printing `{"error": "..."}` objects.
- Follow-up local review found the docs overpromised JSON for Typer parse errors
  and the plan had the previous full-suite count. Wording now limits JSON error
  objects to service/runtime errors after CLI parsing succeeds, and verification
  records the final 249-test full-suite run.
- Follow-up ANSI RED:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py::test_service_json_commands_stay_plain_json_when_console_color_is_enabled tests/test_cli_service_commands.py::test_service_json_errors_stay_plain_json_when_console_color_is_enabled -q`
  failed with all four cases containing ANSI escape sequences from Rich
  rendering under a forced-color console.
- Follow-up ANSI GREEN:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py::test_service_json_commands_stay_plain_json_when_console_color_is_enabled tests/test_cli_service_commands.py::test_service_json_errors_stay_plain_json_when_console_color_is_enabled -q`
  passed with 4 tests after `_print_machine_json()` wrote stdlib JSON directly
  to the console stream.
- CLI service command slice:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q` passed with 230
  tests.
