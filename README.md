# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)

**Keep GPU** is a simple CLI app that keeps your GPUs running.

- 🧾 License: MIT
- 📚 Documentation: https://keepgpu.readthedocs.io

---

Contributions Welcome!

If you have ideas for new features or improvements, feel free to open an issue or submit a pull request.

This project does not yet fully support ROCm GPUs, so any contributions, suggestions, or testing help in that area are especially welcome!

---

## Features

- Simple command-line interface
- Uses PyTorch and `nvidia-smi` to monitor and load GPUs
- Easy to extend for your own keep-alive logic

---

## Installation

```bash
pip install keep-gpu
```

## Usage


```bash
keep-gpu
```

Specify the interval in microseconds between GPU usage checks (default is 300 seconds):
```bash
keep-gpu --interval 100
```

Specify GPU IDs to run on (default is all available GPUs):
```bash
keep-gpu --gpu-ids 0,1,2
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

## Contributors

<!-- google-doc-style-ignore -->
<a href="https://github.com/Wangmerlyn/KeepGPU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Wangmerlyn/KeepGPU" />
</a>
<!-- google-doc-style-resume -->
