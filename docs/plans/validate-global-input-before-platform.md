# Validate Global Inputs Before Platform Probe Plan

## Background

`GlobalGPUController.__init__()` called `get_platform()` before validating local
constructor inputs. That means obviously invalid Python API values such as empty
or duplicate `gpu_ids`, non-positive intervals, invalid busy thresholds, and bad
VRAM values could trigger backend/platform probes first. On GPU hosts, those
probes may initialize NVML, ROCm SMI, torch CUDA, or MPS before returning a
user-correctable input error.

## Goal

Reject invalid local `GlobalGPUController` inputs before platform or hardware
probing, keeping visible-count validation after discovery because it depends on
the current backend/device count.

## Solution

- Add RED constructor tests that monkeypatch `get_platform()` to fail and prove
  invalid local inputs should raise their own validation errors first.
- Reuse existing shared validators for `gpu_ids`, `interval`, and
  `busy_threshold`.
- Reuse `parse_vram_to_elements()` only as a validation step, preserving the
  raw public `vram_to_keep` value that child controllers already normalize.
- Leave CUDA/ROCm/MACM visible-count and platform-specific ID validation after
  backend discovery.
- Document the validation-before-probe contract in `AGENTS.md`, Python/API docs,
  architecture docs, and this plan.

## Tasks

- [x] Add RED test coverage for invalid local inputs before `get_platform()`.
- [x] Move local validation ahead of platform detection.
- [x] Keep visible-count checks after platform/device discovery.
- [x] Update `AGENTS.md`, Python/API/architecture docs, and this plan.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before PR.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

Completed so far:

- Baseline `PYTHONPATH=$PWD/src pytest tests/global_controller -q`:
  `9 passed, 1 skipped`.
- RED focused regression:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py::test_global_controller_validates_local_inputs_before_platform_probe -q`
  failed because all invalid-input cases reached `get_platform()` first.
- GREEN focused regression:
  `PYTHONPATH=$PWD/src pytest tests/global_controller/test_contract.py::test_global_controller_validates_local_inputs_before_platform_probe -q`:
  `10 passed`.
- Global-controller shard: `PYTHONPATH=$PWD/src pytest tests/global_controller -q`:
  `17 passed, 1 skipped`.
- Targeted controller/platform shard:
  `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/global_controller tests/utilities/test_platform_manager.py -q`:
  `34 passed, 6 skipped`.
- Full test suite: `PYTHONPATH=$PWD/src pytest tests -q`:
  `288 passed, 11 skipped`.
- `PYTHONPATH=$PWD/src mkdocs build`: passed. Existing Material for MkDocs
  warning and unnav'd plan notices were emitted.
- `pre-commit run --all-files`: passed.
- `git diff --check && git diff --cached --check`: passed.
