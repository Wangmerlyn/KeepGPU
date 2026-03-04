# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Wangmerlyn/KeepGPU)
[![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Wangmerlyn/KeepGPU)](https://coderabbit.ai/dashboard/gh/Wangmerlyn/KeepGPU)
[![SkillCheck Passed](https://raw.githubusercontent.com/olgasafonova/skillcheck-free/main/skill-check/passed.svg)](https://github.com/olgasafonova/skillcheck-free)

**Keep GPU** keeps shared GPUs from being reclaimed while you prep data, debug, or coordinate multi-stage pipelines. It allocates just enough VRAM and issues lightweight CUDA work so schedulers observe an “active” device—without running a full training job.

- 🧾 License: MIT
- 📚 Docs: https://keepgpu.readthedocs.io

## Why it exists

On many clusters, idle GPUs are reaped or silently shared after a short grace period. The cost of losing your reservation (or discovering another job has taken your card) can dwarf the cost of a tiny keep-alive loop. KeepGPU is a minimal, auditable guardrail:

- **Predictable** – Single-purpose controller with explicit resource knobs (VRAM size, interval, utilization backoff).
- **Polite** – Uses NVML to read utilization and backs off when the GPU is busy.
- **Portable** – Typer/Rich CLI for humans; Python API for orchestrators and notebooks.
- **Observable** – Structured logging and optional file logs for auditing what kept the GPU alive.
- **Power-aware** – Uses intervalled elementwise ops instead of heavy matmul floods to present “busy” utilization while keeping power and thermals lower (see `CudaGPUController._run_relu_batch` for the loop).
- **NVML-backed** – GPU telemetry comes from `nvidia-ml-py` (the `pynvml` module), with optional `rocm-smi` support when you install the `rocm` extra.

## Quick start (CLI)

```bash
pip install keep-gpu

# Hold GPU 0 with 1 GiB VRAM and throttle if utilization exceeds 25%
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60

# Non-blocking mode for agent workflows (auto-starts local service)
keep-gpu start --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
keep-gpu status
```

Open the dashboard while service mode is running:

```text
http://127.0.0.1:8765/
```

### Platform installs at a glance

- **CUDA (example: cu121)**
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch
  pip install keep-gpu
  ```
- **ROCm (example: rocm6.1)**
  ```bash
  pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch
  pip install keep-gpu[rocm]
  ```
- **CPU-only**
  ```bash
  pip install torch
  pip install keep-gpu
  ```

Flags that matter:

- Blocking mode knobs:
  - `--vram` (`1GiB`, `750MB`, or bytes): how much memory to pin.
  - `--interval` (seconds): sleep between keep-alive bursts.
  - `--busy-threshold`: skip work when NVML reports higher utilization.
  - `--gpu-ids`: target a subset; otherwise all visible GPUs are guarded.
- Service mode commands:
  - `keep-gpu serve`: run local service (HTTP + dashboard).
  - `keep-gpu start`: create keep session and return immediately.
  - `keep-gpu status`: inspect active sessions.
  - `keep-gpu stop --job-id <id>` or `keep-gpu stop --all`: release sessions.
  - `keep-gpu list-gpus`: fetch telemetry from local service.

## Embed in Python

```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

with CudaGPUController(rank=0, interval=0.5, vram_to_keep="1GiB", busy_threshold=20):
    preprocess_dataset()   # GPU is marked busy while you run CPU-heavy code

train_model()              # GPU freed after exiting the context
```

Need multiple devices?

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

with GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="750MB", interval=90, busy_threshold=30):
    run_pipeline_stage()
```

## What you get

- Battle-tested keep-alive loop built on PyTorch.
- NVML-based utilization monitoring (by way of `nvidia-ml-py`) to avoid hogging busy GPUs; optional ROCm SMI support by way of `pip install keep-gpu[rocm]`.
- CLI + API parity: same controllers power both code paths.
- Continuous docs + CI: mkdocs + mkdocstrings build in CI to keep guidance up to date.

## For developers

- Install dev extras: `pip install -e ".[dev]"` (add `.[rocm]` if you need ROCm SMI).
- Fast CUDA checks: `pytest tests/cuda_controller tests/global_controller tests/utilities/test_platform_manager.py tests/test_cli_thresholds.py`
- ROCm-only tests carry `@pytest.mark.rocm`; run with `pytest --run-rocm tests/rocm_controller`.
- Markers: `rocm` (needs ROCm stack) and `large_memory` (opt-in locally).

### MCP and service API

- Start a simple JSON-RPC server on stdin/stdout (default):
  ```bash
  keep-gpu-mcp-server
  ```
- Or expose it over HTTP (JSON-RPC + REST + dashboard):
  ```bash
  keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
  ```
- JSON-RPC request example:
  ```json
  {"id": 1, "method": "start_keep", "params": {"gpu_ids": [0], "vram": "512MB", "interval": 60, "busy_threshold": 20}}
  ```
- REST examples:
  ```bash
  curl http://127.0.0.1:8765/health
  curl http://127.0.0.1:8765/api/sessions
  ```
- Methods: `start_keep`, `stop_keep` (optional `job_id`, default stops all), `status` (optional `job_id`), `list_gpus` (basic info).
- Dashboard: `http://127.0.0.1:8765/`
- Minimal client config (stdio MCP):
  ```yaml
  servers:
    keepgpu:
      command: ["keep-gpu-mcp-server"]
      adapter: stdio
  ```
- Minimal client config (HTTP MCP):
  ```yaml
  servers:
    keepgpu:
      url: http://127.0.0.1:8765/
      adapter: http
  ```
- Remote/SSH tunnel example (HTTP):
  ```bash
  keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
  ```
  Client config (replace hostname/tunnel as needed):
  ```yaml
  servers:
    keepgpu:
      url: http://gpu-box.example.com:8765/
      adapter: http
  ```
  For untrusted networks, put the server behind your own auth/reverse-proxy or
  tunnel by way of SSH (for example, `ssh -L 8765:localhost:8765 gpu-box`).

## Contributing

Contributions are welcome—especially around ROCm support, platform fallbacks, and scheduler-specific recipes. Open an issue or PR if you hit edge cases on your cluster.
See [docs/contributing.md](docs/contributing.md) for dev setup, test commands, and PR tips.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

## Contributors

<!-- google-doc-style-ignore -->
<a href="https://github.com/Wangmerlyn/KeepGPU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Wangmerlyn/KeepGPU" />
</a>
<!-- google-doc-style-resume -->

## 📖 Citation

If you find **KeepGPU** useful in your research or work, please cite it as:

```bibtex
@software{Wangmerlyn_KeepGPU_2025,
  author       = {Wang, Siyuan and Shi, Yaorui and Liu, Yida and Yin, Yuqi},
  title        = {KeepGPU: a simple CLI app that keeps your GPUs running},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17129114},
  url          = {https://github.com/Wangmerlyn/KeepGPU},
  note         = {GitHub repository},
  keywords     = {ai, hpc, gpu, cluster, cuda, torch, debug}
}
