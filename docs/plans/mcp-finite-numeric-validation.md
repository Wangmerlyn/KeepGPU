# Finite Numeric Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject oversized public numeric values as validation errors before they can overflow interval or VRAM parsing paths.

**Architecture:** Keep public interval validation centralized in `src/keep_gpu/utilities/session_config.py` and public VRAM parsing centralized in `src/keep_gpu/utilities/humanized_input.py`. JSON-RPC, REST, MCP tools, CLI, and Python controllers should inherit the same finite, bounded contract through those shared helpers instead of each surface handling overflow differently.

**Tech Stack:** Python, pytest, KeepGPU MCP/JSON-RPC/REST service, MkDocs.

---

## Background

The previous non-finite interval fix rejects `NaN` and infinity, but some huge
finite-looking public values still fail before they can become normal validation
errors:

- `validate_interval(10**1000)` can raise `OverflowError` while calling
  `math.isfinite()`.
- `parse_vram_to_elements(10**1000)` can raise `OverflowError` while converting
  public byte counts through `float(value)`.
- Oversized digit strings such as `"999...GiB"` can also overflow during
  `float(value)` conversion.

Because `_validate_public_session_input()` maps only `TypeError` and
`ValueError` into `SessionInputError`, those overflows can surface as JSON-RPC
`-32603`, HTTP `500`, or MCP tool execution errors that look like internal
failures. They are user-correctable input errors and should stay in the public
validation lane.

## Solution

- Add RED tests for oversized interval and VRAM inputs at the shared utility
  layer, direct JSON-RPC layer, and REST prevalidation layer.
- Add explicit public bounds for interval seconds and VRAM byte-equivalent
  requests. Intervals are capped by Python's runtime wait limit, VRAM
  byte-equivalent requests are capped at 1 PiB, and oversized values raise
  `ValueError`.
- Avoid unchecked `float()` conversion for public integer byte counts. Integer
  byte counts can be converted to float32 element counts with integer division.
- Catch numeric conversion `OverflowError` inside `humanized_input.py` and raise
  `ValueError` with a user-facing message.
- Parse decimal VRAM strings exactly enough to enforce the 1 PiB ceiling before
  conversion to tensor elements.
- Add a narrow `_validate_public_session_input()` defense-in-depth catch for
  `OverflowError` so public validators cannot leak overflow as internal server
  failures if a future helper regresses.
- Update `AGENTS.md` and user docs to state that public numeric session inputs
  must be finite, positive, and bounded.

## Tasks

### Task 1: Add RED utility tests

**Files:**
- Modify: `tests/utilities/test_session_config.py`
- Modify: `tests/utilities/test_humanized_input.py`

- [x] Add a test proving `validate_interval(10**1000)` raises `ValueError`
      with a public validation message instead of `OverflowError`.
- [x] Add tests proving huge integer and huge digit-string VRAM inputs raise
      `ValueError` from `parse_vram_to_elements()`/`parse_size()`.
- [x] Run:
      `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/utilities/test_humanized_input.py -q`
      and record the expected RED failures.

### Task 2: Add RED service boundary tests

**Files:**
- Modify: `tests/mcp/test_server.py`
- Modify: `tests/mcp/test_http_api.py`

- [x] Add JSON-RPC tests proving huge `interval`, huge integer `vram`, and huge
      string `vram` return `-32602 Invalid params` without creating sessions.
- [x] Add REST tests proving the same oversized values return HTTP `400` before
      `list_gpus()` or controller startup.
- [x] Run:
      `PYTHONPATH=$PWD/src pytest tests/mcp/test_server.py tests/mcp/test_http_api.py -q -k 'huge or oversized'`
      and record the expected RED failures.

### Task 3: Implement shared validation

**Files:**
- Modify: `src/keep_gpu/utilities/session_config.py`
- Modify: `src/keep_gpu/utilities/humanized_input.py`
- Modify: `src/keep_gpu/mcp/server.py`

- [x] Define an explicit maximum public interval and reject values above it with
      `ValueError`.
- [x] Define an explicit maximum public VRAM byte-equivalent request and reject
      values above it with `ValueError`.
- [x] Convert public integer byte counts to element counts with integer
      arithmetic.
- [x] Validate digit-only byte strings before `int()` conversion so Python's
      integer digit limit cannot leak into user-facing errors.
- [x] Validate decimal VRAM strings against the 1 PiB ceiling before element
      conversion.
- [x] Validate unit-suffixed decimal VRAM strings against the per-unit ceiling
      before multiplication can round just-over-limit values back to the cap.
- [x] Validate blocking CLI VRAM before torch import, telemetry probing, or
      controller startup.
- [x] Widen MCP `tools/list` schema so `vram` accepts strings or integer bytes
      and advertises the 1 PiB cap.
- [x] Wrap string numeric conversion overflow in `ValueError`.
- [x] Map public `OverflowError` through `SessionInputError` as a final guard.
- [x] Run targeted utility and MCP/HTTP tests until green.

### Task 4: Update documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/guides/mcp.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/concepts/architecture.md`
- Modify: this plan

- [x] Document that public `interval` values are finite positive seconds capped
      to a reasonable keep-loop bound.
- [x] Document that public `vram` values remain bytes for integers and
      digit-only strings, but must be finite and bounded.
- [x] Keep wording consistent across CLI, REST, JSON-RPC, MCP, and Python API
      references.

### Task 5: Verify, review, PR, and merge

- [x] Run targeted tests:
      `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/utilities/test_humanized_input.py tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_service_commands.py tests/global_controller/test_contract.py -q`
- [x] Run full tests:
      `PYTHONPATH=$PWD/src pytest tests -q`
- [x] Run docs build:
      `PYTHONPATH=$PWD/src mkdocs build`
- [x] Run pre-commit:
      `pre-commit run --all-files`
- [x] Run `git diff --check`.
- [x] Run local subagent spec and code-quality review; resolve all must-fix
      findings.
- [ ] Commit with `fix(validation): reject oversized numeric session inputs`.
- [ ] Push, open a PR titled
      `[validation,mcp] fix: reject oversized numeric session inputs`.
- [ ] Resolve all GitHub and local review comments, wait for green checks, squash
      merge, pull `main`, and clean the branch/worktree.

## Verification

- Baseline before edits:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/mcp/test_server.py tests/mcp/test_http_api.py -q`,
  `150 passed`.
- RED:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/utilities/test_humanized_input.py tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_service_commands.py tests/global_controller/test_contract.py -q -k 'oversized or huge or local_inputs_before_auto_start or validates_local_inputs_before_platform_probe'`,
  `14 failed, 12 passed, 222 deselected`; failures showed `OverflowError`,
  JSON-RPC `-32603`, HTTP `500`, and uncaught CLI/controller overflows.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/utilities/test_humanized_input.py tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_service_commands.py tests/global_controller/test_contract.py -q -k 'oversized or huge or local_inputs_before_auto_start or validates_local_inputs_before_platform_probe'`,
  `26 passed, 222 deselected`.
- Touched-module shard:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_session_config.py tests/utilities/test_humanized_input.py tests/mcp/test_server.py tests/mcp/test_http_api.py tests/test_cli_service_commands.py tests/global_controller/test_contract.py -q`,
  `248 passed` before review fixes, then `258 passed` after addressing local
  review findings for blocking CLI, MCP schema, digit-only string limits, and
  exact decimal VRAM caps, then `259 passed` after the final unit-suffixed
  decimal cap fix.
- Utility rerun after Black formatting:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_humanized_input.py tests/utilities/test_session_config.py -q`,
  `41 passed`.
- Review-fix RED:
  `PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py::test_blocking_mode_rejects_invalid_vram_without_raw_exception tests/mcp/test_server.py::test_mcp_tools_list_exposes_keepgpu_actions tests/mcp/test_server.py::test_mcp_tools_call_rejects_oversized_integer_vram_as_tool_error tests/utilities/test_humanized_input.py tests/utilities/test_session_config.py -q`,
  `5 failed, 47 passed`; failures showed blocking CLI raw VRAM errors, MCP
  string-only `vram` schema, and decimal strings just over 1 PiB passing.
- Review-fix GREEN:
  Same command, `52 passed`.
- Unit-decimal RED:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_humanized_input.py::test_parse_size_rejects_unit_decimal_string_above_public_maximum -q`,
  `1 failed`; `1048576.0000000000000000000000001GiB` rounded to the 1 PiB cap.
- Unit-decimal GREEN:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_humanized_input.py -q`,
  `11 passed`.
- Manual review probes:
  `parse_size(str(PUBLIC_VRAM_MAX_BYTES))` accepted the max,
  `parse_size(str(PUBLIC_VRAM_MAX_BYTES + 1))`,
  `parse_size(f"{PUBLIC_VRAM_MAX_BYTES}.1")`, and `parse_size("9" * 4301)`
  raised `ValueError: vram must be no more than 1 PiB`; MCP `tools/list`
  exposed `vram` as `["string", "integer"]` with the 1 PiB maximum; blocking
  CLI printed `Error: vram must be no more than 1 PiB`.
- Local subagent review:
  initial spec review found blocking CLI VRAM validation, MCP schema parity,
  Decimal precision, and docs table gaps; code-quality review found the
  4301-digit string message edge. All findings were fixed with tests.
  Final spec and code-quality re-reviews reported no remaining issues and
  marked the branch ready to PR.
- Full suite after all edits:
  `PYTHONPATH=$PWD/src pytest tests -q`,
  `349 passed, 11 skipped` before review fixes; `359 passed, 11 skipped` after
  blocking CLI, MCP schema, exact decimal cap, and digit-limit cleanup; `360
  passed, 11 skipped` after the final unit-suffixed decimal cap fix.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build`, passed on final post-review rerun with
  the known Material/MkDocs notice and unlisted-plan warnings.
- Pre-commit:
  `pre-commit run --all-files`, passed on final post-review rerun after Black
  reformatted `src/keep_gpu/utilities/humanized_input.py` in the first run.
- Diff whitespace:
  `git diff --check`, passed on final post-review rerun.
