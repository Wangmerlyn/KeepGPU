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
    Public VRAM byte-equivalent values are capped at 1 PiB so absurd requests
    fail as validation errors instead of overflow failures.
  - `--interval` (finite positive seconds, including fractional values): sleep between keep-alive bursts.
    Values above the Python runtime wait limit are rejected.
  - `--busy-threshold`: defaults to `25`; `0..100` skips work when telemetry reports higher utilization or cannot report utilization; `-1` disables utilization backoff.
  - `--gpu-ids`: target unique non-negative visible device ordinals after user-supplied visibility filtering (`CUDA_VISIBLE_DEVICES` on CUDA, `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` on ROCm). Omit the option to guard all visible GPUs. Empty, whitespace-only, duplicate, or out-of-range selections are invalid, and startup fails if no GPUs resolve.
- Service mode commands:
  - `keep-gpu serve`: run local service (HTTP + dashboard).
  - `keep-gpu start`: create keep session and return immediately.
    Service endpoint flags are validated locally: `--host` must be a DNS
    hostname or IPv4 address, and `--port` must be in `1..65535`.
  - `keep-gpu status`: inspect tracked sessions, including in-progress or failed releases.
  - `keep-gpu stop --job-id <id>` or `keep-gpu stop --all`: release sessions.
    The two stop targets are mutually exclusive; passing both returns a JSON
    error before contacting the service.
    Explicit `--job-id` values for `status` and `stop` must be non-empty
    URL-path-safe IDs; invalid values return JSON errors before any RPC or
    stop-all fallback.
  - `keep-gpu service-stop`: stop the ownership-verified auto-started local daemon.
  - `keep-gpu list-gpus`: fetch telemetry from local service. Each listed
    `id` is the visible ordinal accepted by `--gpu-ids`; optional
    `physical_id`/`uuid` fields are metadata only. CUDA and ROCm devices are
    listed only when Torch can select the same visible ordinal; NVML-only or
    unselectable ROCm inventory is hidden instead of producing unusable
    `gpu_ids`.
  - `status`, `stop`, and `list-gpus` print structured JSON objects, including
    `{"error": "..."}` for service/runtime errors after CLI parsing succeeds,
    that tools such as `jq` can parse directly. Malformed JSON-RPC service
    envelopes and invalid service endpoints are reported as those JSON error
    objects instead of empty success results or tracebacks.

## Embed in Python

```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

with CudaGPUController(rank=0, interval=0.5, vram_to_keep="1GiB", busy_threshold=20):
    preprocess_dataset()   # GPU is marked busy while you run CPU-heavy code

train_model()              # GPU freed after exiting the context
```

Direct CUDA/ROCm controller `rank` values are visible ordinals in the current
process environment and are validated during construction. Non-integer,
negative, or out-of-range ranks fail before KeepGPU creates a device handle or
starts a keep worker.

Need multiple devices?

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

with GlobalGPUController(gpu_ids=[0, 1], vram_to_keep="750MB", interval=90, busy_threshold=30):
    run_pipeline_stage()
```

Pass `gpu_ids=None` to use all visible GPUs. Explicit values are visible device
ordinals, not physical NVML/ROCm SMI IDs. Passing an empty, duplicate, or
out-of-range list is invalid, and startup raises an error if discovery resolves
to zero devices.

## What you get

- Battle-tested keep-alive loop built on PyTorch.
- NVML-based utilization monitoring (by way of `nvidia-ml-py`) to avoid hogging busy GPUs; optional ROCm SMI support by way of `pip install keep-gpu[rocm]`. Public entry points default `busy_threshold` to `25`. Valid values are `-1` or `0..100`; if utilization is unavailable and the threshold is non-negative, KeepGPU sleeps before allocating keep tensors or running compute. CUDA utilization checks use visible CUDA ordinals, so with `CUDA_VISIBLE_DEVICES=3,5`, rank `1` reads NVML telemetry for physical GPU `5`; malformed, duplicate/equivalent, ambiguous, or out-of-range CUDA masks are treated as unavailable telemetry before partial handle lookup. ROCm utilization similarly resolves visible ranks through `ROCR_VISIBLE_DEVICES` and one matching `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm SMI. Ambiguous mappings are treated as unavailable telemetry.
- CLI + API parity: same controllers power both code paths.
- Continuous docs + CI: mkdocs + mkdocstrings build in CI to keep guidance up to date.

## For developers

- Install dev extras: `pip install -e ".[dev]"` (add `.[rocm]` if you need ROCm SMI).
- Fast CUDA checks: `pytest tests/cuda_controller tests/global_controller tests/utilities/test_platform_manager.py tests/test_cli_thresholds.py`
- ROCm visibility tests use mocks and run without hardware; ROCm-only hardware tests carry `@pytest.mark.rocm` and run with `pytest --run-rocm tests/rocm_controller`.
- Markers: `rocm` (needs ROCm stack) and `large_memory` (opt-in locally).

### MCP and service API

- Start an MCP server on stdin/stdout (default):
  ```bash
  keep-gpu-mcp-server
  ```
- Or expose it over HTTP (JSON-RPC + REST + dashboard):
  ```bash
  keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
  ```
- MCP clients use the standard `initialize`, `tools/list`, and `tools/call`
  protocol methods over stdio. KeepGPU exposes four tools: `start_keep`,
  `stop_keep`, `status`, and `list_gpus`.
- Legacy direct JSON-RPC method calls remain supported for scripts:
  ```json
  {"id": 1, "method": "start_keep", "params": {"gpu_ids": [0], "vram": "512MB", "interval": 60, "busy_threshold": 20}}
  ```
- Stdio stdout is reserved for JSON protocol messages; diagnostics and logs are
  written to stderr.
- HTTP mode is KeepGPU's local JSON-RPC/REST/dashboard service. It accepts the
  same JSON-RPC messages at `/rpc`, but it is not a Streamable HTTP MCP
  endpoint.
- Successful legacy direct JSON-RPC responses use a KeepGPU direct-method
  envelope with `jsonrpc: "2.0"`, the matching request `id`, and an object
  `result`.
- REST examples:
  ```bash
  curl http://127.0.0.1:8765/health
  curl http://127.0.0.1:8765/api/sessions
  ```
- Methods: `start_keep`, `stop_keep` (optional `job_id`, default stops all), `status` (optional `job_id`), `list_gpus` (basic info). REST session creation accepts a JSON object body, not arrays or scalar values. Omitting `gpu_ids` uses all visible GPUs, and omitting `busy_threshold` uses the eco-safe default `25`; explicit values must be unique visible ordinals in the service process environment. `list_gpus` returns those same start-compatible ordinals as `id`/`visible_id`; `physical_id` and `uuid` are informational metadata, not valid substitutes for `gpu_ids`. On CUDA, NVML records are returned only when Torch CUDA can address the same visible ordinal set, so NVML-only devices are not advertised as startable. On ROCm, records are returned only when Torch can select the visible ordinal; nullable memory fields mean memory telemetry is unavailable after successful selection. Empty, duplicate, or out-of-range lists are invalid and startup fails if no GPUs resolve. Public numeric session inputs must stay finite and bounded: interval values are positive seconds, including fractional seconds, capped by the runtime wait limit, and VRAM byte-equivalent values are capped at 1 PiB. Custom `job_id` values must be unique across active and starting sessions, and only `null`/omitted means generated or all-sessions; custom IDs must be non-empty strings containing only letters, digits, `.`, `_`, `-`, or `~`. Status responses include reserved jobs as `state="starting"` while controller startup is still in progress.
- Supported REST route/method failures remain machine-readable: validation
  errors use JSON `400` responses, unknown API routes use JSON `404`, and
  unexpected service/runtime failures use JSON `500` instead of closing the
  connection without a response.
  The dashboard reads those structured payloads and displays `error.message`
  instead of the raw JSON body.
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
- Dashboard GPU cards show the visible ordinal to type into the start form first,
  with physical/vendor metadata shown only as secondary context.
- Dashboard cards mirror lifecycle state so a retained session shows
  `Releasing` or `Release failed` instead of being presented as a fully active
  keepalive.
- Dashboard utilization summaries ignore unavailable readings and show `n/a`
  when no finite readings exist; per-GPU cards also omit the utilization fill
  for unavailable telemetry so unknown readings are not presented as idle.
- Dashboard: `http://127.0.0.1:8765/`
- **Mac M series limitations:**
  - GPU utilization monitoring is not available on macOS.
  - The default `busy_threshold=25` keeps MPS in conservative sleep-only mode; set `busy_threshold=-1` to opt into unconditional keepalive compute.
  - `list-gpus` reports best-effort MPS memory counters and `null` for unsupported telemetry fields.
- Minimal client config (stdio MCP):
  ```yaml
  servers:
    keepgpu:
      command: ["keep-gpu-mcp-server"]
      adapter: stdio
  ```
- Remote/SSH tunnel example (HTTP):
  ```bash
  keep-gpu-mcp-server --mode http --host 0.0.0.0 --port 8765
  ```
  Use `http://gpu-box.example.com:8765/` for the dashboard and
  `http://gpu-box.example.com:8765/rpc` for JSON-RPC scripts.
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
