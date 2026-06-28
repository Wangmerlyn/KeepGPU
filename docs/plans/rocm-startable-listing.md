# ROCm Startable Listing Plan

## Background

`list_gpus` exposes `id` as the visible ordinal users can pass to `gpu_ids`.
ROCm controller startup, however, fails synchronously when
`torch.cuda.set_device(rank)` fails. The ROCm listing path currently catches
that same failure together with nullable memory probing, then still emits a
record with `memory_total=None` and `memory_used=None`.

## Goal

Keep ROCm GPU inventory honest: do not advertise a visible ordinal that the
ROCm controller cannot select during startup.

## Solution

- Add a regression test where ROCm Torch reports two visible devices but
  `torch.cuda.set_device(1)` fails.
- Make `_query_rocm()` treat `set_device(idx)` failure as an unstartable visible
  ordinal and skip that record.
- Keep `mem_get_info()` best-effort after a successful `set_device()`, so
  nullable memory still represents telemetry unavailability rather than startup
  impossibility.
- Update docs and `AGENTS.md` to state that ROCm listings follow the same
  startability contract as keep/start APIs.

## Tasks

- [x] Add RED regression coverage in `tests/utilities/test_gpu_info.py`.
- [x] Implement the minimal `_query_rocm()` control-flow split.
- [x] Update `docs/concepts/architecture.md` and `AGENTS.md`.
- [x] Align README, CLI, Python API, MCP, and REST docs with the ROCm
  start-compatible listing contract.
- [x] Run targeted GPU info and ROCm/controller tests.
- [x] Run broad verification, pre-commit, docs build, and diff check.
- [ ] Create a PR, run local subagent review, resolve review feedback, and
  squash merge only after hosted checks/comments are clean.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py::test_get_gpu_info_rocm_hides_unstartable_visible_ordinals -q`
  failed because listing returned `[0, 1]` instead of `[0]`.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py::test_get_gpu_info_rocm_hides_unstartable_visible_ordinals -q`
  passed.
- Focused:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py tests/rocm_controller/test_rocm_backoff.py tests/rocm_controller/test_rocm_utilization.py -q`
  passed with 56 passed, 1 skipped.
- Targeted:
  `PYTHONPATH=$PWD/src pytest tests/utilities/test_gpu_info.py tests/rocm_controller tests/global_controller/test_contract.py -q`
  passed with 97 passed, 1 skipped.
- Full:
  `PYTHONPATH=$PWD/src pytest tests -q` passed with 576 passed, 11 skipped.
- Docs:
  `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material warning
  and unnav'd plan-page notices.
- Hygiene:
  `pre-commit run --all-files` and `git diff --check` passed.
