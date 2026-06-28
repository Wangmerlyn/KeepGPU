# Global Python VRAM Default Plan

## Background

`GlobalGPUController` defaulted `vram_to_keep` to `10 * (2**30)` bytes, while
the CLI, service, JSON-RPC, REST, and MCP public surfaces default to `1GiB`.
Direct Python users who omitted `vram_to_keep` therefore reserved ten times more
VRAM per GPU than the rest of KeepGPU.

## Goal

Align the Python global controller with the low-power public default so omitted
VRAM settings reserve `1GiB` per GPU by default.

## Solution

- Add a signature-level contract test for `GlobalGPUController.vram_to_keep`.
- Change the default to `"1GiB"` while preserving explicit integer-byte and
  human-readable string support.
- Document the shared low-power default in API docs, Python recipes, and
  `AGENTS.md`.

## Todo

- [x] Run the global-controller baseline.
- [x] Add the failing default-contract test.
- [x] Verify the new test fails on current behavior.
- [x] Implement the one-line default change.
- [x] Verify the focused test and global contract shard pass.
- [x] Run broader tests, docs build, pre-commit, and local subagent review.
- [ ] Open a PR, resolve review comments, squash merge, and clean up the branch.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py -q`
- `PYTHONPATH=$PWD/src pytest tests/global_controller -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
