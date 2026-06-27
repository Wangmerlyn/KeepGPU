# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Wangmerlyn/KeepGPU)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Wangmerlyn/KeepGPU?utm_source=oss&utm_medium=github&utm_campaign=Wangmerlyn%2FKeepGPU&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)
[![SkillCheck Passed](https://raw.githubusercontent.com/olgasafonova/skillcheck-free/main/skill-check/passed.svg)](https://github.com/olgasafonova/skillcheck-free)

**Keep GPU** keeps shared GPUs from being reclaimed while you prep data, debug, or coordinate multi-stage pipelines. It allocates just enough VRAM and issues lightweight CUDA work so schedulers observe an “active” device—without running a full training job.

- 🧾 License: MIT
- 📚 Docs: https://keepgpu.readthedocs.io

## Why it exists

On many clusters, idle GPUs are reaped or silently shared after a short grace period. The cost of losing your reservation (or discovering another job has taken your card) can dwarf the cost of a tiny keep-alive loop. KeepGPU is a minimal, auditable guardrail:

- **Predictable** – Single-purpose controller with explicit resource knobs (VRAM size, interval, utilization backoff).
- **Polite** – Uses telemetry to read utilization and backs off when the GPU is busy or utilization is unavailable.
- **Portable** – Typer/Rich CLI for humans; Python API for orchestrators and notebooks.
- **Observable** – Structured logging and optional file logs for auditing what kept the GPU alive.
- **Power-aware** – Uses intervalled elementwise ops instead of heavy matmul floods to present “busy” utilization while keeping power and thermals lower (see `CudaGPUController._run_relu_batch` for the loop).
- **Telemetry-aware** – GPU telemetry comes from `nvidia-ml-py` (the `pynvml` module), optional `rocm-smi`, and best-effort MPS memory counters on Mac M series.

## Quick start (CLI)

```bash
pip install keep-gpu

# Hold GPU 0 with 1 GiB VRAM and throttle if utilization exceeds 25%
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60

# Non-blocking mode for agent workflows (auto-starts local service)
keep-gpu start --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
keep-gpu status
keep-gpu stop --all
keep-gpu service-stop
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
- **Mac M series (M1/M2/M3/M4)**
  ```bash
  pip install torch
  pip install keep-gpu[macm]
  ```
  Uses Metal Performance Shaders (MPS) backend on Apple Silicon.

Flags that matter:

- Blocking mode knobs:
  - `--vram` (`1GiB`, `750MB`, or bare bytes like `1073741824`): how much memory to pin.
  - `--interval` (positive seconds): sleep between keep-alive bursts.
  - `--busy-threshold`: `0..100` skips work when telemetry reports higher utilization or cannot report utilization; `-1` disables utilization backoff.
  - `--gpu-ids`: target unique non-negative visible device ordinals after any user-supplied `CUDA_VISIBLE_DEVICES` filtering; otherwise all visible GPUs are guarded. Empty, duplicate, or out-of-range selections are invalid, and startup fails if no GPUs resolve.
- Service mode commands:
  - `keep-gpu serve`: run local service (HTTP + dashboard).
  - `keep-gpu start`: create keep session and return immediately.
  - `keep-gpu status`: inspect tracked sessions, including in-progress or failed releases.
  - `keep-gpu stop --job-id <id>` or `keep-gpu stop --all`: release sessions.
  - `keep-gpu service-stop`: stop the ownership-verified auto-started local daemon.
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

Pass `gpu_ids=None` to use all visible GPUs. Explicit values are visible device
ordinals, not physical NVML IDs. Passing an empty, duplicate, or out-of-range
list is invalid, and startup raises an error if discovery resolves to zero
devices.

## What you get

- Battle-tested keep-alive loop built on PyTorch.
- NVML-based utilization monitoring (by way of `nvidia-ml-py`) to avoid hogging busy GPUs; optional ROCm SMI support by way of `pip install keep-gpu[rocm]`. Valid `busy_threshold` values are `-1` or `0..100`; if utilization is unavailable and the threshold is non-negative, KeepGPU sleeps for that cycle instead of running compute. CUDA utilization checks use visible CUDA ordinals, so with `CUDA_VISIBLE_DEVICES=3,5`, rank `1` reads NVML telemetry for physical GPU `5`; ambiguous mappings are treated as unavailable telemetry.
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
- Methods: `start_keep`, `stop_keep` (optional `job_id`, default stops all), `status` (optional `job_id`), `list_gpus` (basic info). Omitting `gpu_ids` uses all visible GPUs, but explicit values must be unique visible ordinals in the service process environment. Empty, duplicate, or out-of-range lists are invalid and startup fails if no GPUs resolve. Custom `job_id` values must be unique across active and starting sessions, and only `null`/omitted means generated or all-sessions; custom IDs must be non-empty strings containing only letters, digits, `.`, `_`, `-`, or `~`. Status responses include reserved jobs as `state="starting"` while controller startup is still in progress.
- Stop responses distinguish completed cleanup from partial cleanup:
  `stopped` means released, while `timed_out` sessions remain visible as
  `stopping` until background cleanup completes and `failed` sessions remain
  visible with `state` and `last_error`.
- Status and stop requests both account for in-progress starts: status reports
  them as `starting`, and stop waits for startup to settle so a session is not
  reported as missing or skipped by stop-all.
- Stop-all only covers sessions active or already starting when that request
  begins; later concurrent starts belong to a later stop request.
- Stop-all releases independent sessions concurrently and reports outcomes in
  deterministic snapshot order with the same `stopped`, `timed_out`, `failed`,
  and `errors` fields.
- Dashboard cards mirror that lifecycle state so a retained session shows
  `Releasing` or `Release failed` instead of being presented as a fully active
  keepalive.
- Dashboard: `http://127.0.0.1:8765/`
- **Mac M series limitations:**
  - GPU utilization monitoring is not available on macOS.
  - Non-negative `busy_threshold` values therefore keep MPS in conservative sleep-only mode; set `busy_threshold=-1` to opt into unconditional keepalive compute.
  - `list-gpus` reports best-effort MPS memory counters and `null` for unsupported telemetry fields.
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
