# KeepGPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)

KeepGPU is a small, polite GPU keeper for shared machines: reserve the VRAM you
ask for, back off when the device is busy, and release cleanly when you are
done.

It works with CUDA, ROCm/HIP, and Apple Silicon MPS when PyTorch can see the
target device.

## Quick Start

```console
pip install keep-gpu
keep-gpu --gpu-ids 0 --vram 1GiB --busy-threshold 25 --interval 60
```

The command blocks until you press `Ctrl+C`, then releases the reserved memory.
MPS utilization telemetry is unavailable; use `--busy-threshold -1` only
when you intentionally want unconditional keepalive compute.

## Learn More

- [Documentation](https://keepgpu.readthedocs.io/en/latest/) covers the CLI,
  Python API, service dashboard, JSON-RPC, REST, and MCP server.
- [Getting Started](https://keepgpu.readthedocs.io/en/latest/getting-started/)
  has install options and the first hardware sanity check.
- [Contributing](https://keepgpu.readthedocs.io/en/latest/contributing/) covers
  local setup, tests, docs builds, and PR guidance.
- [Citation](https://keepgpu.readthedocs.io/en/latest/citation/) has the BibTeX
  entry.
