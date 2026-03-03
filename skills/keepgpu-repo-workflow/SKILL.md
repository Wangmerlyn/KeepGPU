---
name: keepgpu-repo-workflow
description: Implement, test, and document changes in the KeepGPU repository while preserving CLI, Python API, and MCP parity, platform boundaries, and GPU-safe behavior on no-GPU CI runners. Use when tasks request code changes, bug fixes, refactors, tests, docs updates, releases, or pull-request work in this repository; do not use for unrelated repositories or generic Python questions.
---

# KeepGPU Repository Workflow

Follow this workflow to make reliable, review-ready changes in KeepGPU.

## Prerequisites

- Work from repository root.
- Confirm branch starts from latest `main`.
- Keep user-facing text and comments in English.

## Repository map

- CLI entrypoint: `src/keep_gpu/cli.py`
- Single-GPU controllers: `src/keep_gpu/single_gpu_controller/`
- Global controller orchestration: `src/keep_gpu/global_gpu_controller/`
- MCP server: `src/keep_gpu/mcp/server.py`
- Platform probing: `src/keep_gpu/utilities/platform_manager.py`
- GPU telemetry helpers: `src/keep_gpu/utilities/gpu_info.py`

## Implementation rules

1. Start from a new branch and keep diffs focused.
2. For non-trivial work, create or update a plan in `docs/plans/` with background, goal, solution, and todo items.
3. Keep platform detection centralized in `platform_manager.py`; do not spread platform branching across unrelated modules.
4. Keep telemetry logic in `gpu_info.py` or related utility modules.
5. Preserve controller flow: global controller orchestrates per-GPU controllers; single-GPU controllers handle keep/release loops.
6. Keep CUDA telemetry on `nvidia-ml-py` (`pynvml` module import) and keep ROCm support optional with graceful failure on non-ROCm hosts.
7. Update docs when behavior changes for CLI flags, controllers, platform support, or MCP methods.
8. Keep commits narrow and use `type(scope): summary` commit messages.

## Validation order

Run the smallest relevant checks first, then broader checks.

1. Targeted tests for changed modules.
2. Broader tests if changes touch shared logic.
3. `pre-commit run --all-files` before push.
4. `mkdocs build` when docs are changed.

Preferred targeted test commands:

```bash
pytest tests/cuda_controller tests/global_controller tests/utilities/test_platform_manager.py
pytest tests -k threshold
pytest tests/mcp tests/utilities/test_gpu_info.py
```

ROCm-only command (run only on ROCm-capable machines):

```bash
pytest --run-rocm tests/rocm_controller
```

## Output format

Return results in this structure:

```markdown
## KeepGPU Task Result

### Changes
- <file-level summary>

### Validation
- Targeted tests: <pass/fail + command>
- Broader checks: <pass/fail + command>
- Docs build: <pass/fail or not run>
- Pre-commit: <pass/fail>

### Risks
- <known caveat or "none">
```

## Example

User request: "Add a new CLI flag for busy-threshold defaults and update docs."

Execution pattern:

1. Update `src/keep_gpu/cli.py` and related controller wiring.
2. Add or update tests in `tests/test_cli_thresholds.py` and relevant controller tests.
3. Update user docs and README sections for the flag behavior.
4. Run targeted tests, then `pre-commit run --all-files`, then `mkdocs build`.
5. Prepare a focused commit and PR summary.

## Limitations

- GPU hardware is often unavailable in CI; guard hardware-dependent logic and tests.
- Do not bump versions, retag releases, or alter release metadata unless explicitly requested.
