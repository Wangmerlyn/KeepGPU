# CUDA NVML Startable Listing Plan

## Background

`get_gpu_info()` uses NVML before Torch on CUDA systems so list responses can
include utilization and vendor metadata. `GlobalGPUController`, however, starts
CUDA sessions with visible Torch ordinals from `torch.cuda.device_count()`.

When NVML can see CUDA devices but Torch CUDA is unavailable, reports zero
devices, or reports a visible count that does not match a CUDA visibility mask,
`list_gpus` can advertise IDs that users cannot successfully pass as
`gpu_ids`.

## Goal

Keep public GPU listings start-compatible: CUDA NVML records should be returned
only when Torch CUDA can address the same visible ordinal set. If NVML cannot be
trusted for the startable set, fall back to Torch listing when possible; if Torch
cannot start CUDA devices, list no CUDA devices.

## Solution

- Add RED coverage for NVML-visible but Torch-unstartable CUDA environments.
- Require a trustworthy Torch CUDA visible count before returning NVML CUDA
  records.
- Require the resolved CUDA visible-token count to match Torch's visible count.
- Preserve HIP/ROCm precedence so HIP Torch builds never fall back to NVML CUDA
  records.
- Update agent and user docs to state that listed CUDA IDs are start-compatible
  Torch visible ordinals, not NVML-only inventory.

## Tasks

- [x] Add RED tests for Torch CUDA unavailable, zero count, device-count failure,
      and CUDA visibility/Torch count mismatch.
- [x] Implement the minimal NVML guard and Torch fallback behavior.
- [x] Update `AGENTS.md`, README, CLI reference, MCP guide, and Python/API docs.
- [x] Run focused GPU info tests.
- [x] Run targeted GPU/MCP and global-controller-adjacent tests.
- [x] Run full tests, docs build, hooks, and whitespace checks.
- [x] Request local subagent code review and resolve findings before PR.

## Verification Log

- RED:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py -q` failed with
  four focused failures:
  Torch CUDA unavailable, Torch count zero, Torch count failure, and CUDA
  visible-token/Torch count mismatch still returned NVML records.
- GREEN focused:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py -q` passed with
  24 tests and 1 skipped.
- MCP/GPU targeted:
  `PYTHONPATH=$PWD/src pytest tests/mcp tests/utilities/test_gpu_info.py -q`
  passed with 193 tests and 1 skipped.
- Global-controller targeted:
  `PYTHONPATH=$PWD/src pytest tests/global_controller tests/utilities/test_gpu_info.py -q`
  passed with 76 tests and 2 skipped.
- Local review follow-up:
  added RED tests for two reviewer findings:
  matching NVML/Torch counts still listed a CUDA device when
  `torch.cuda.set_device(0)` failed, and non-HIP fallback could label a CUDA
  Torch fallback as ROCm when `rocm_smi` was importable. Both focused
  regressions now pass.
- Final review follow-up:
  added a RED test proving non-HIP listing must not fall through to ROCm SMI
  when both NVML and Torch reject an unstartable CUDA ordinal. Removed the
  non-HIP ROCm SMI fallback and updated architecture/docstring wording.
- Review follow-up targeted:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py -q` passed with
  27 tests and 1 skipped after the final review follow-up.
- Review follow-up MCP/GPU targeted:
  `PYTHONPATH=$PWD/src pytest tests/mcp tests/utilities/test_gpu_info.py -q`
  passed with 196 tests and 1 skipped after the final review follow-up.
- Review follow-up global-controller targeted:
  `PYTHONPATH=$PWD/src pytest tests/global_controller tests/utilities/test_gpu_info.py -q`
  passed with 79 tests and 2 skipped after the final review follow-up.
- Full suite:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 568 tests and 11 skipped
  after the local review follow-up.
- Docs build:
  `PYTHONPATH=$PWD/src mkdocs build` passed. It emitted the existing Material
  for MkDocs warning and unnav'd docs notices, including this plan page.
- Hooks:
  `pre-commit run --all-files` passed.
- Whitespace:
  `git diff --check` passed.
