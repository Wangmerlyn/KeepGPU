# Fix CLI JSON Output Plan

## Background

`keep-gpu status`, `keep-gpu stop`, and `keep-gpu list-gpus` pass
`json.dumps(result)` as `data` to Rich's `console.print_json()`. Rich treats
that string as data and emits a top-level JSON string, so callers must decode
twice and tools such as `jq` cannot index the output directly.

## Goal

Emit structured JSON objects from service CLI commands so one `json.loads()` or
one shell JSON tool invocation sees the expected object.

## Solution

- Add RED CLI tests for `status`, `stop --job-id`, `stop --all`, and
  `list-gpus` that require a single JSON decode to return a dict.
- Update the existing stop-all fallback test to single-decode the command
  output.
- Print decoded result objects with `console.print_json(data=result)`.
- Document that these CLI commands emit directly parseable JSON objects.

## Tasks

- [x] Add RED tests for single-decode structured JSON output.
- [x] Implement minimal CLI output changes.
- [x] Update `AGENTS.md`, CLI docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- Baseline: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with 34 tests.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_status_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_job_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_all_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_list_gpus_outputs_single_decoded_json_object tests/test_cli_service_commands.py::test_stop_all_fallback_force_stops_managed_daemon -q`
  failed with all five outputs decoding once to strings instead of dicts.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with 38 tests.
- `PYTHONPATH=$PWD/src pytest tests -q` passed with 248 tests and 11 skipped.
- `PYTHONPATH=$PWD/src mkdocs build` passed with existing Material/MkDocs and
  unlisted-plan warnings.
- `pre-commit run --all-files` passed.
- `git diff --check` passed.
