# Contributing & Development

Thanks for helping improve KeepGPU! This page collects the key commands and
expectations so you can get productive quickly and avoid surprises in CI.

## Setup

- Clone and install dev extras:
  ```bash
  git clone https://github.com/Wangmerlyn/KeepGPU.git
  cd KeepGPU
  pip install -e ".[dev]"       # add .[rocm] if you need ROCm SMI
  ```
- Ensure you have the right torch build for your platform (CUDA/ROCm/CPU).
- Optional: install `nvidia-ml-py` (CUDA) or `rocm-smi` (ROCm) for telemetry.

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
- MCP + utilities:
  ```bash
  pytest tests/mcp tests/utilities/test_gpu_info.py
  ```
- All tests honor markers `rocm` and `large_memory`; avoid enabling
  `large_memory` in CI.

## Lint/format

- Run pre-commit hooks locally before pushing:
  ```bash
  pre-commit run --all-files
  ```

## MCP server (experimental)

- Start: `keep-gpu-mcp-server` (stdin/stdout JSON-RPC)
- Methods: `start_keep`, `stop_keep`, `status`, `list_gpus`
- Example request:
  ```json
  {"id":1,"method":"start_keep","params":{"gpu_ids":[0],"vram":"512MB","interval":60,"busy_threshold":20}}
  ```

## Pull requests

- Keep changesets focused; small commits are welcome.
- Add/adjust tests for new behavior; skip GPU-specific tests in CI by way of markers.
- Update docs/README when behavior or interfaces change.
- Stick to the existing style (Typer CLI, Rich logging) and keep code paths
  simple—avoid over-engineering.

## Support

- Issues/PRs: https://github.com/Wangmerlyn/KeepGPU
- Code of Conduct: see `CODE_OF_CONDUCT.rst`
