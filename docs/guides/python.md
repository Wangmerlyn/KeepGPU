# Python API Recipes

Embed KeepGPU directly inside orchestration scripts so GPUs stay warm only for
the stages you choose.

## Keep a single GPU while you do CPU work

```python
from keep_gpu.single_gpu_controller.cuda_gpu_controller import CudaGPUController

def preprocess_shards():
    ...

with CudaGPUController(rank=0, interval=0.5, vram_to_keep="1.5GiB"):
    preprocess_shards()          # GPU 0 is marked “busy” the whole time

train_model()                     # GPU memory is released automatically
```

- `rank` matches the visible device index after environment filtering.
  CUDA utilization backoff resolves that visible rank through
  `CUDA_VISIBLE_DEVICES` before querying NVML; for example, with
  `CUDA_VISIBLE_DEVICES=3,5`, rank `1` reads physical GPU `5`. Duplicate or
  ambiguous CUDA masks are treated as unavailable telemetry. ROCm utilization
  resolves `ROCR_VISIBLE_DEVICES` as the base mask and one matching
  `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm
  SMI. If a mapping cannot be resolved, utilization is treated as unavailable.
- `interval` is the finite positive pause between keep-alive bursts inside the background thread.
- `vram_to_keep` accepts integer bytes or human-readable strings (`parse_size` handles it).

## Start/stop manually

Need more control? Call `keep()` and `release()` yourself.

```python
ctrl = CudaGPUController(rank=1, interval=1.0, vram_to_keep=1_073_741_824)

ctrl.keep()
run_cpu_bound_stage()
ctrl.release()
```

The controller spins up a daemon thread. Repeated `keep()` calls are idempotent
and simply warn if the worker is already running.

## Guard multiple GPUs with a single context

```python
from keep_gpu.global_gpu_controller.global_gpu_controller import GlobalGPUController

gpu_ids = [0, 2, 3]              # use None to cover all visible GPUs

with GlobalGPUController(
    gpu_ids=gpu_ids,
    interval=90,
    vram_to_keep="512MB",
    busy_threshold=40,
):
    run_pipeline_controller()
```

- Each `CudaGPUController` runs in its own thread.
- `gpu_ids=None` means all visible GPUs. Explicit values are visible device
  ordinals after CUDA or ROCm visibility filtering. Empty, duplicate, or
  out-of-range lists are invalid, and startup raises `ValueError` if the
  resolved selection contains zero devices.
- `busy_threshold` defaults to `25` and accepts `-1` or a percentage in
  `0..100`. Non-negative thresholds throttle the keep-alive loop when
  utilization spikes. When utilization telemetry is unavailable, non-negative
  thresholds sleep before allocating the keep tensor or running compute; use
  `busy_threshold=-1` only for explicit unconditional keepalive work.
- `release()` uses threads too, so all GPUs free up quickly.

## Combine with schedulers or callbacks

```python
def wait_until_dataset_ready(ctrl, poll_fn):
    ctrl.keep()
    while not poll_fn():
        time.sleep(30)
    ctrl.release()


def main():
    ctrl = CudaGPUController(rank=0, interval=0.2, vram_to_keep="2GiB")
    wait_until_dataset_ready(ctrl, lambda: Path("/tmp/done").exists())
    launch_training_job()
```

- Encapsulate the keep/release lifecycle in helper functions so you do not forget
  to free the GPU if a stage fails.
- Wrap logic in `try/finally` or `contextlib.ExitStack` if you perform multiple
  guarded operations sequentially.

## Troubleshooting

- **OOM during `keep()`** – Lower `vram_to_keep`. KeepGPU logs the failure and
  retries after `interval` seconds, but repeated OOMs usually indicate another
  process is already using the GPU.
- **Controllers never stop** – Ensure you call `release()` even when exceptions
  occur. Context managers are the safest way to guarantee cleanup.
- **Need a CPU-only fallback?** – `GlobalGPUController` supports CUDA, ROCm, and
  Mac M backends when the matching PyTorch/runtime stack is available. In
  CPU-only environments, catch the startup error and skip the guard logic for
  that run.
