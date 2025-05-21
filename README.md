# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Build Status](https://img.shields.io/travis/Wangmerlyn/KeepGPU.svg)](https://travis-ci.com/Wangmerlyn/KeepGPU)
[![Docs Status](https://readthedocs.org/projects/keep-gpu/badge/?version=latest)](https://keep-gpu.readthedocs.io/en/latest/?version=latest)
[![PyUp Updates](https://pyup.io/repos/github/Wangmerlyn/keep-gpu/shield.svg)](https://pyup.io/repos/github/Wangmerlyn/keep-gpu/)

**Keep GPU** is a simple CLI app that keeps your GPUs running.

- 🧾 License: MIT
- 📚 Documentation: https://keep-gpu.readthedocs.io

---

## Features

- Simple command-line interface
- Uses PyTorch and `nvidia-smi` to monitor and load GPUs
- Easy to extend for your own keep-alive logic

---

## TODO ✅

- [ ] Add more CLI args (e.g. `--gpu-id`, `--gpu-ids`, `--gpu-keep-threshold`, `--gpu-keep-time`, `--gpu-keep-vram-usage`)
- [ ] Add documentation
- [ ] Add importable Python functions

---

## Installation

```bash
pip install keep-gpu
```
## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.