# Contributing & Development

Thanks for helping improve KeepGPU! This page collects the key commands and
expectations so you can get productive quickly and avoid surprises in CI.

## Setup

- Clone and install dev extras:
  ```bash
  git clone https://github.com/Wangmerlyn/KeepGPU.git
  cd KeepGPU
  pip install -e ".[dev]"
  ```
- Ensure you have the right torch build for your platform (CUDA/ROCm/CPU).
- Telemetry note: CUDA uses the base `nvidia-ml-py` dependency. ROCm SMI comes
  from the ROCm/system stack as `rocm_smi`; KeepGPU handles it gracefully when
  unavailable.

## Tests

- Fast CUDA suite:
  ```bash
  pytest tests/cuda_controller tests/global_controller \
    tests/utilities/test_platform_manager.py tests/test_cli_thresholds.py
  ```
- ROCm-only tests are marked `rocm` and skipped by default; run with:
  ```bash
  pytest --run-rocm tests/rocm_controller
  ```
- Heavy VRAM tests are marked `large_memory` and skipped by default; run only
  on a machine where the allocation is acceptable:
  ```bash
  pytest --run-large-memory -m large_memory
  ```
- MCP + utilities:
  ```bash
  pytest tests/mcp tests/utilities/test_gpu_info.py
  ```
- Avoid enabling `large_memory` in CI.
- Keep broad validation matrices with the utility or controller that owns the
  contract; interface tests should use representative smoke cases plus
  side-effect guards instead of repeating every edge case.
- When changing CUDA visibility telemetry, cover numeric and UUID
  `CUDA_VISIBLE_DEVICES` masks, including NVML UUID string/bytes lookup
  differences.

## Lint/format

- Run pre-commit hooks locally before pushing:
  ```bash
  pre-commit run --all-files
  ```
- Use the documented `pyproject.toml`, CI, MkDocs, and dashboard package
  commands as the source of truth; the old `setup.py`/Sphinx command scaffold is
  intentionally not part of the repository.
- Keep Ruff settings in `pyproject.toml`; do not add a standalone `ruff.toml`
  unless the full configuration is intentionally migrated there.
- Keep pre-commit CI lean: install the `pre-commit` runner only, and let hooks
  provision their own tool environments instead of installing KeepGPU runtime
  dependencies.
- Keep Python CI installs explicit: do not add a root `requirements.txt`
  fallback. Use `pyproject.toml` for runtime/test dependencies and
  `docs/requirements.txt` for documentation builds.
- Keep build metadata lean: list directly used third-party build/runtime
  distributions, do not rely on transitive dependencies, and do not list
  Python standard library modules such as `argparse`.
- Keep release artifacts lean: avoid shipping the test suite in sdists by
  default, and enumerate required runtime assets instead of using broad package
  data wildcards.
- Keep package metadata warning-free with modern SPDX license strings and keep
  the supported Python version floor aligned with build-backend requirements.
- Keep cosmetic logging helpers optional. Console logging must work through the
  Python standard library when packages such as `colorlog` are absent.
- Keep package metadata such as `requires-python` aligned with the documented
  supported Python versions.
- Keep project URLs in package metadata pointing to live repository pages.
- Keep metadata tests self-contained for simple checks; avoid importing parser
  libraries that are only available through transitive test dependencies.

## Docs

- Install documentation dependencies once:
  ```bash
  pip install -r docs/requirements.txt
  ```
- Keep directly invoked documentation tools and configured MkDocs extensions in
  `docs/requirements.txt`; for example, `mkdocs build` and configured
  `pymdownx.*` extensions should not depend on theme packages transitively.
- Live preview:
  ```bash
  mkdocs serve
  ```
- Build the static site:
  ```bash
  mkdocs build
  ```
- API reference pages are resolved from the checkout's `src/` tree, so docs-only
  builds do not need `pip install .`.
- Internal agent plans and skill reports under `docs/plans/` and `docs/skills/`
  stay in the repository but are excluded from the published MkDocs site.
- Keep README as a concise front door. Put full citation metadata in
  `docs/citation.md` and link to it from README.

## MCP server (experimental)

- Start: `keep-gpu-mcp-server` (stdin/stdout JSON-RPC)
- HTTP option: `keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765`
- Methods: `start_keep`, `stop_keep`, `status`, `list_gpus`
- Example request:
  ```json
  {"id":1,"method":"start_keep","params":{"gpu_ids":[0],"vram":"512MB","interval":60,"busy_threshold":20}}
  ```
- Remote tip: for shared clusters, prefer HTTP behind your own auth/reverse-proxy
  or tunnel with SSH (`ssh -L 8765:localhost:8765 gpu-box`), then point your MCP
  client at `http://127.0.0.1:8765/`.

## Pull requests

- Keep changesets focused; small commits are welcome.
- For parallel agent work, branch from the latest `main` and place worktrees
  under `.worktrees/`:
  ```bash
  git fetch origin
  git worktree add .worktrees/codex/my-fix -b codex/my-fix origin/main
  ```
- Add/adjust tests for new behavior; skip GPU-specific tests in CI by way of markers.
- Update docs/README when behavior or interfaces change.
- Run a local code review pass before merging; squash merge only after all
  review comments are resolved.
- Stick to the existing style (Typer CLI, stdlib logging with optional color
  helpers) and keep code paths simple—avoid over-engineering.

## Support

- Issues/PRs: https://github.com/Wangmerlyn/KeepGPU
- Code of Conduct: see `CODE_OF_CONDUCT.rst`
