# KeepGPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)

KeepGPU is a small, polite GPU keeper for shared machines. It reserves the VRAM
you ask for, backs off when devices are busy, and releases cleanly when you are
done.

## Quick Start

```console
pip install keep-gpu
keep-gpu --gpu-ids 0 --vram 1GiB --interval 60
```

Press `Ctrl+C` to release. CUDA, ROCm/HIP, and Apple Silicon MPS are supported
when PyTorch can see the target device.

## Documentation

- [Documentation](https://keepgpu.readthedocs.io/en/latest/) is the full guide hub.
- [Getting Started](https://keepgpu.readthedocs.io/en/latest/getting-started/) covers install options and the first hardware check.
- [Citation](https://keepgpu.readthedocs.io/en/latest/citation/) has the BibTeX entry.
