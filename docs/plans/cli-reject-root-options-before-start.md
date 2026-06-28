# CLI Reject Root Options Before Start Plan

## Background

The root callback owns blocking-mode options such as `--gpu-ids`, `--vram`,
`--busy-threshold`/`--util-threshold`, hidden `--threshold`, and `--interval`.
When a service subcommand is invoked, the callback currently returns before
checking whether those root options were explicitly supplied. A command such as
`keep-gpu --gpu-ids 0 start` therefore starts a service session with
`gpu_ids=None`, which means all visible GPUs.

## Solution

Use Click's parameter-source tracking in the root callback. If a subcommand is
being invoked and any blocking-mode root option came from the command line,
raise a clear error telling the user to put service options after the
subcommand, for example `keep-gpu start --gpu-ids 0`. Defaults that were not
explicitly supplied must remain ignored so normal service commands keep working.

## Tasks

- [x] Add a failing service-command regression test for root blocking options
      supplied before `start`.
- [x] Confirm the test fails before implementation.
- [x] Implement the minimal root-callback guard before service startup or RPC.
- [x] Preserve blocking mode when no subcommand is invoked.
- [x] Add a non-`start` service-command regression so the broader guard stays
      intentional.
- [x] Address hosted review by accepting raw `"COMMANDLINE"` parameter-source
      values as well as Click enum members.
- [x] Add the root/subcommand option-placement guideline to `AGENTS.md`.
- [x] Run targeted tests and `git diff --check`.
- [x] Commit with `fix(cli): reject root options before service commands`.

## Verification Notes

- RED: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_start_rejects_root_blocking_options_before_service_calls -q` failed before implementation with 6 failing parameter cases because the command exited successfully instead of rejecting root options.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_start_rejects_root_blocking_options_before_service_calls -q` passed with 6 cases after the guard.
- Review-polish check: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_status_rejects_root_blocking_options_before_service_rpc -q` passed after adding the non-`start` service-command regression.
- Hosted-review RED/GREEN: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_root_option_source_helper_accepts_raw_commandline_value -q` failed before the fallback and passed after the helper accepted raw string sources.
- Final targeted suite: `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py tests/test_cli_thresholds.py -q` passed with 126 tests.
- Diff hygiene: `git diff --check` passed.
