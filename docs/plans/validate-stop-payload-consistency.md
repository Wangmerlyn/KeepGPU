# Validate Stop Payload Consistency

## Background

The CLI validates `stop_keep` service results before rendering JSON output or
triggering daemon stop side effects. It already checks the required field types,
but it does not reject internally inconsistent additive outcomes.

## Goal

Reject malformed `stop_keep` results when a job id appears more than once, appears
in more than one outcome list, or when the `errors` mapping does not correspond
exactly to the `failed` list.

## Solution

Keep the check in `src/keep_gpu/cli.py` alongside the existing method-specific
payload validation. The server contract remains unchanged: `stopped`,
`timed_out`, and `failed` are disjoint job-id lists, and `errors` contains one
string error for each failed job only.

## Todo

- [x] Add focused RED tests in `tests/test_cli_service_commands.py` for duplicate
      job ids, cross-list job ids, missing failed errors, and stray error keys.
- [x] Update `_validate_stop_keep_result()` with minimal consistency checks.
- [x] Clarify the CLI contract in `AGENTS.md` and user docs where stop payloads
      are described.
- [x] Run targeted CLI stop tests, broader CLI service tests, full pytest,
      pre-commit, and docs build before opening the PR.

## Verification

- RED: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'stop_job_rejects_inconsistent_outcome_payloads or stop_all_rejects_inconsistent_payload_before_stopping_daemon'`
  failed with six exit-code assertions before the validator change.
- GREEN: the same command passed with `6 passed, 184 deselected`.
- Focused: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'stop_job_outputs_single_decoded_json_object or stop_job_rejects_malformed_payloads or stop_job_rejects_malformed_job_id_lists_and_errors or stop_job_rejects_inconsistent_outcome_payloads or stop_all_outputs_single_decoded_json_object or stop_all_rejects_malformed_payload or stop_all_rejects_inconsistent_payload_before_stopping_daemon or service_stop_rejects_malformed_stop_keep_before_stopping_daemon'`
  passed with `22 passed, 168 deselected`.
- CLI service: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q`
  passed with `190 passed`.
- Full suite: `PYTHONPATH=$PWD/src pytest tests -q` passed with
  `691 passed, 11 skipped`.
- Quality: `pre-commit run --all-files` passed.
- Docs: plain `mkdocs build` could not import `keep_gpu`; `PYTHONPATH=$PWD/src mkdocs build`
  passed.
