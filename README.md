# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Wangmerlyn/KeepGPU)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Wangmerlyn/KeepGPU?utm_source=oss&utm_medium=github&utm_campaign=Wangmerlyn%2FKeepGPU&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)
[![SkillCheck Passed](https://raw.githubusercontent.com/olgasafonova/skillcheck-free/main/skill-check/passed.svg)](https://github.com/olgasafonova/skillcheck-free)

KeepGPU keeps shared GPUs reserved while you prepare data, debug, or coordinate pipelines by holding a small, polite keep-alive workload instead of a full training job.

## Why KeepGPU

- Guard interactive or staged work from idle GPU reclaim policies.
- Reserve only the VRAM you ask for, with a low-power `1GiB` default.
- Back off when utilization shows the device is busy, unless you explicitly opt into unconditional keepalive.
- Use the same session contract from the CLI, Python controllers, and MCP service.

## Quick Start

Install the package:

```bash
pip install keep-gpu
```

Run a blocking keep-alive in the current shell:

```bash
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
```

Use service mode when a workflow needs the command to return immediately:

```bash
keep-gpu start --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
keep-gpu status
keep-gpu stop --all
keep-gpu service-stop
```

## Python

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

with GlobalGPUController(gpu_ids=[0], vram_to_keep="1GiB", interval=60):
    preprocess_dataset()
    run_pipeline_stage()
```

## Service, Dashboard, and MCP

Service mode provides local session control for agents and long-running shells. Start it explicitly with `keep-gpu serve`, or let `keep-gpu start` auto-start it for local use.

Open the dashboard while service mode is running:

```text
http://127.0.0.1:8765/
```

The MCP server is available as `keep-gpu-mcp-server` over stdio, with optional HTTP mode for browser and local service access. See [MCP and Service API](docs/guides/mcp.md) for protocol and deployment details.

## Documentation

- [Getting Started](docs/getting-started.md)
- [CLI Guide](docs/guides/cli.md)
- [Python Guide](docs/guides/python.md)
- [MCP and Service API](docs/guides/mcp.md)
- [CLI Reference](docs/reference/cli.md)
- [Python API Reference](docs/reference/api.md)
- [Architecture](docs/concepts/architecture.md)

## Contributing

Contributions are welcome, especially around platform fallbacks and scheduler-specific recipes. See [Contributing](docs/contributing.md) for setup, tests, and PR guidance.

## Citation

If you find KeepGPU useful in your research or work, please cite it as:

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
```
