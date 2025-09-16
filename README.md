# Keep GPU

[![PyPI Version](https://img.shields.io/pypi/v/keep-gpu.svg)](https://pypi.python.org/pypi/keep-gpu)
[![Docs Status](https://readthedocs.org/projects/keepgpu/badge/?version=latest)](https://keepgpu.readthedocs.io/en/latest/?version=latest)
[![DOI](https://zenodo.org/badge/987167271.svg)](https://doi.org/10.5281/zenodo.17129114)

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

### Use keep-gpu as a cli tool

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

### Use keep-gpu api in your code

Non-blocking gpu keeping logic with `CudaGPUController`:
```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
ctrl = CudaGPUController(rank=0, interval=0.5)
# occupy GPU while you do CPU-only work
# this is non-blocking
ctrl.keep()
dataset.process()
ctrl.release()        # give GPU memory back
model.train_start()   # now run real GPU training
```

Use `CudaGPUController` as a context manager:
```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController
with CudaGPUController(rank=0, interval=0.5):
    dataset.process()  # GPU occupied inside this block
model.train_start()    # GPU free after exiting block
```

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
