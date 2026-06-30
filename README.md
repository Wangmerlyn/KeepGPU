# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Wangmerlyn/KeepGPU)
[![CodeRabbit Reviews](https://img.shields.io/coderabbit/prs/github/Wangmerlyn/KeepGPU?utm_source=oss&utm_medium=github&utm_campaign=Wangmerlyn%2FKeepGPU&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)](https://coderabbit.ai)

KeepGPU keeps shared GPUs reserved while you prepare data, debug, or coordinate pipelines by holding a small, polite keep-alive workload instead of a full training job.

## Why KeepGPU

- Guard interactive or staged work from idle GPU reclaim policies.
- Reserve only the VRAM you ask for, with a low-power `1GiB` default.
- Back off when utilization shows the device is busy, unless you explicitly opt into unconditional keepalive.

## Quick Start

```bash
pip install keep-gpu
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
```

The command blocks until you press `Ctrl+C`, then releases the reserved memory.
For non-blocking sessions, dashboard controls, and agent integrations, use the
guides below.

## Choose an Interface

| Need | Start here |
| --- | --- |
| First install and sanity check | [Getting Started](https://keepgpu.readthedocs.io/en/latest/getting-started/) |
| Shell workflows and service mode | [CLI Guide](https://keepgpu.readthedocs.io/en/latest/guides/cli/) |
| Python context managers/controllers | [Python Guide](https://keepgpu.readthedocs.io/en/latest/guides/python/) |
| Dashboard, REST, JSON-RPC, and MCP | [MCP and Service API](https://keepgpu.readthedocs.io/en/latest/guides/mcp/) |
| Full command list | [CLI Reference](https://keepgpu.readthedocs.io/en/latest/reference/cli/) |
| Public Python API | [Python API Reference](https://keepgpu.readthedocs.io/en/latest/reference/api/) |
| Design and lifecycle model | [Architecture](https://keepgpu.readthedocs.io/en/latest/concepts/architecture/) |

## Documentation

The published documentation is available at
[keepgpu.readthedocs.io](https://keepgpu.readthedocs.io/). The same pages live
under [`docs/`](https://github.com/Wangmerlyn/KeepGPU/tree/main/docs) in this
repository.

## Contributing

Contributions are welcome, especially around platform fallbacks and
scheduler-specific recipes. See
[Contributing](https://keepgpu.readthedocs.io/en/latest/contributing/) for
setup, tests, and PR guidance.

## Citation

If KeepGPU helps your research or operations, see
[Citation](https://keepgpu.readthedocs.io/en/latest/citation/) for the BibTeX
entry.
