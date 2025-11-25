# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)

**Keep GPU** keeps shared GPUs from being reclaimed while you prep data, debug, or coordinate multi-stage pipelines. It allocates just enough VRAM and issues lightweight CUDA work so schedulers observe an “active” device—without running a full training job.

- 🧾 License: MIT
- 📚 Docs: https://keepgpu.readthedocs.io

## Why it exists

On many clusters, idle GPUs are reaped or silently shared after a short grace period. The cost of losing your reservation (or discovering another job has taken your card) can dwarf the cost of a tiny keep-alive loop. KeepGPU is a minimal, auditable guardrail:

- **Predictable** – Single-purpose controller with explicit resource knobs (VRAM size, interval, utilization backoff).
- **Polite** – Uses NVML to read utilization and backs off when the GPU is busy.
- **Portable** – Typer/Rich CLI for humans; Python API for orchestrators and notebooks.
- **Observable** – Structured logging and optional file logs for auditing what kept the GPU alive.
- **Power-aware** – Uses intervalled elementwise ops instead of heavy matmul floods to present “busy” utilization while keeping power and thermals lower (see `CudaGPUController._run_mat_batch` for the loop).
- **NVML-backed** – GPU telemetry comes from `nvidia-ml-py` (the `pynvml` module), with optional `rocm-smi` support when you install the `rocm` extra.

## Quick start (CLI)

```bash
pip install keep-gpu

# Hold GPU 0 with 1 GiB VRAM and throttle if utilization exceeds 25%
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
```

Flags that matter:

- `--vram` (`1GiB`, `750MB`, or bytes): how much memory to pin.
- `--interval` (seconds): sleep between keep-alive bursts.
- `--busy-threshold`: skip work when NVML reports higher utilization.
- `--gpu-ids`: target a subset; otherwise all visible GPUs are guarded.

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
- NVML-based utilization monitoring (by way of `nvidia-ml-py`) to avoid hogging busy GPUs; optional ROCm SMI support via `pip install keep-gpu[rocm]`.
- CLI + API parity: same controllers power both code paths.
- Continuous docs + CI: mkdocs + mkdocstrings build in CI to keep guidance up to date.

## Contributing

Contributions are welcome—especially around ROCm support, platform fallbacks, and scheduler-specific recipes. Open an issue or PR if you hit edge cases on your cluster.

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
