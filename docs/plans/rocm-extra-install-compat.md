# ROCm Extra Install Compatibility Plan

## Background

The `rocm` optional dependency currently advertises `rocm-smi`, but
`rocm-smi` is not resolvable from PyPI. Installing `keep-gpu[rocm]` can
therefore fail before users reach KeepGPU's runtime handling. The code already
treats `import rocm_smi` as optional and degrades gracefully when the ROCm
system stack does not provide it.

## Goal

Keep the `rocm` extra name install-compatible for existing users while avoiding
any dependency on a non-PyPI ROCm SMI distribution.

## Solution

- Add a metadata regression test that asserts the `rocm` extra remains declared
  and does not include `rocm-smi`.
- Change `pyproject.toml` to keep `rocm = []`.
- Update README, user docs, contributor docs, architecture notes, and agent
  guardrails so they say ROCm SMI comes from the ROCm/system stack and missing
  `rocm_smi` is handled gracefully.

## Todo

- [x] Add the focused metadata regression test.
- [x] Run the metadata test red before changing package metadata.
- [x] Remove `rocm-smi` from the `rocm` extra.
- [x] Update ROCm install and telemetry documentation.
- [x] Run targeted tests, package build/dry-run install, docs build,
      `git diff --check`, and pre-commit when feasible.
- [x] Commit the focused fix.

## Verification

Planned commands:

- `python -m pip index versions rocm-smi`
- `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q`
- `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/utilities/test_gpu_info.py -q`
- `python -m build`
- `python -m pip install --dry-run dist/keep_gpu-*.whl[rocm]`
- `PYTHONPATH=$PWD/src mkdocs build`
- `git diff --check`
- `pre-commit run --all-files`

Completed so far:

- Root-cause check: `python -m pip index versions rocm-smi` failed with
  `ERROR: No matching distribution found for rocm-smi`.
- RED: `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q` failed
  because `extras["rocm"]` contained `["rocm-smi"]`.
- GREEN: `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q`
  passed with `1 passed`.
- ROCm/import shard:
  `PYTHONPATH=$PWD/src pytest tests/rocm_controller tests/utilities/test_gpu_info.py -q`
  passed with `77 passed, 1 skipped`.
- Package build: `python -m build --outdir /tmp/keepgpu-rocm-extra-dist.kbwkum`
  built the sdist and wheel successfully. It emitted the existing setuptools
  deprecation warning for `project.license` as a TOML table.
- Install compatibility:
  `python -m pip install --dry-run '/tmp/keepgpu-rocm-extra-dist.kbwkum/keep_gpu-0.5.1-py3-none-any.whl[rocm]'`
  succeeded and did not attempt to resolve `rocm-smi`.
- Docs: `PYTHONPATH=$PWD/src mkdocs build` passed with the existing Material
  warning about MkDocs 2.0.
- Hygiene: `git diff --check` passed.
- Pre-commit: `pre-commit run --all-files` passed.
