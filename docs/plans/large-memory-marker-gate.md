# Large Memory Marker Gate

## Background

KeepGPU documents `large_memory` tests as opt-in because they may allocate
large VRAM. The pytest configuration registered the marker but did not skip it
by default, so the marked CUDA allocation test could run whenever CUDA was
available.

## Goal

Keep the repository small and friendly for ordinary development and CI by making heavy
VRAM tests opt-in with an explicit `--run-large-memory` flag.

## Solution

- Add a pytest `--run-large-memory` option beside the existing `--run-rocm` and
  `--run-macm` gates.
- Skip tests marked `large_memory` unless the flag is supplied.
- Add a hardware-free pytester regression test for both the default skip path
  and the explicit opt-in path.
- Update README, contributor docs, and AGENTS guidance with the exact command.

## Validation

- `PYTHONPATH=$PWD/src pytest tests/test_pytest_marker_gates.py -q`
- `PYTHONPATH=$PWD/src pytest -q -m large_memory tests/cuda_controller/test_2_32pow_elements.py -rs`
- `PYTHONPATH=$PWD/src pytest --collect-only -q -m large_memory tests`
- `pre-commit run --all-files`
