# Busy-Threshold CLI Validation Plan

## Background

`keep-gpu start --busy-threshold abc` currently fails in Typer's integer parser
before KeepGPU can run its shared CLI validator. That produces generic usage
text instead of the project error message:
`busy_threshold must be -1 or an integer between 0 and 100`.

`--interval` and `--port` already accept raw command-line strings at the Typer
boundary and then normalize through KeepGPU validators before service startup or
RPC side effects.

## Goal

Make non-integer busy-threshold CLI values fail through KeepGPU's shared
validation path, with no service auto-start or RPC calls and no Typer usage
text.

## Solution

- Add RED coverage for service-mode `start --busy-threshold abc`.
- Add focused root-mode coverage for `keep-gpu --busy-threshold abc`.
- Preserve the existing guard that rejects blocking-mode root options before
  service subcommands, including malformed root busy-threshold values.
- Change the Typer boundary to accept raw busy-threshold input and parse it in
  `_validate_cli_busy_threshold()`.
- Keep valid explicit values and defaults behavior unchanged.

## Todo

- [x] Add failing regression tests.
- [x] Record the RED failure.
- [x] Implement the minimal parser/annotation fix.
- [x] Update `AGENTS.md` with the raw numeric option validation contract.
- [x] Run targeted tests, docs build, pre-commit, and diff checks.
- [x] Commit the focused change.

## Verification

- RED:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'non_integer_busy_threshold or invalid_root_busy_threshold'`
  failed with 3 failures. Each case exited with Typer code `2` instead of the
  expected project-handled code `1`.
- GREEN focused:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'non_integer_busy_threshold or invalid_root_busy_threshold'`
  passed with `3 passed, 206 deselected`.
- GREEN CLI shard:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q` passed with
  `209 passed`.
- Final focused regression:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q -k 'non_integer_busy_threshold or invalid_root_busy_threshold'`
  passed with `3 passed, 206 deselected`.
- Final CLI service-command shard:
  `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q` passed with
  `209 passed`.
- Final docs build:
  `PYTHONPATH=src mkdocs build --strict` passed. It emitted the existing
  Material for MkDocs warning about future MkDocs 2 compatibility.
- Final pre-commit:
  `pre-commit run --all-files --show-diff-on-failure` passed all hooks.
- Final whitespace check:
  `git diff --check` passed.
