# Visible GPU ID Validation Plan

## Background

`keep-gpu --gpu-ids` is documented as selecting a subset of visible CUDA
ordinals. The blocking CLI currently mutates `CUDA_VISIBLE_DEVICES` to the
provided IDs and still passes those same IDs to `GlobalGPUController`. For a
selection such as `--gpu-ids 3`, that can turn physical GPU `3` into visible
ordinal `0` while the controller still tries to use rank `3`.

`GlobalGPUController` also accepts explicit CUDA/ROCm `gpu_ids` without checking
that they are within `torch.cuda.device_count()`, so invalid visible ordinals can
reach per-device controllers and fail later in a background thread.

## Goal

Keep public `gpu_ids` semantics as visible ordinals, fail invalid selections
before starting workers, and avoid hidden environment mutation that can make
KeepGPU reserve the wrong device or waste power.

## Design

- Do not rewrite `CUDA_VISIBLE_DEVICES` inside blocking CLI mode. Users who need
  physical-device filtering should set that environment variable before running
  KeepGPU.
- Validate explicit CUDA/ROCm `gpu_ids` against the current visible device
  count in `GlobalGPUController` before constructing per-GPU controllers.
- Keep `gpu_ids=None` behavior unchanged: use all visible CUDA/ROCm devices, and
  fail clearly if that resolves to zero devices.
- Update docs and `AGENTS.md` so future code treats `gpu_ids` as visible
  ordinals across CLI, Python API, MCP, and dashboard surfaces.

## Todo

- [x] Add failing global-controller tests for explicit CUDA/ROCm IDs outside
      the visible device count.
- [x] Add a failing CLI blocking-mode test proving `--gpu-ids` does not mutate
      `CUDA_VISIBLE_DEVICES` and passes visible ordinals through unchanged.
- [x] Implement early visible-ID validation in
      `src/keep_gpu/global_gpu_controller/global_gpu_controller.py`.
- [x] Remove blocking CLI `CUDA_VISIBLE_DEVICES` mutation in
      `src/keep_gpu/cli.py`.
- [x] Update `AGENTS.md`, README, CLI/Python docs, and API reference with the
      visible-ordinal contract.
- [x] Run targeted CLI/global-controller tests, full tests, docs build,
      pre-commit, and local subagent review.
- [ ] Open the PR, wait for PR checks/reviews, resolve every required comment,
      squash merge, and clean up the worktree.
