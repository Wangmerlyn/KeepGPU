# Python API Recipes

Embed KeepGPU directly inside orchestration scripts so GPUs stay warm only for
the stages you choose.

For global sessions and telemetry helpers, public CUDA IDs are Torch-startable
visible ordinals. NVML may add utilization and vendor metadata, but NVML-only
devices are not exposed as `gpu_ids` targets when Torch CUDA cannot start them.

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
  Direct CUDA/ROCm controllers validate this value during construction: it must
  be a plain integer in the current visible range, and invalid ranks fail before
  a device handle or keep worker is created.
  CUDA utilization backoff resolves that visible rank through
  `CUDA_VISIBLE_DEVICES` before querying NVML; for example, with
  `CUDA_VISIBLE_DEVICES=3,5`, rank `1` reads physical GPU `5`. Malformed,
  duplicate/equivalent, ambiguous, or out-of-range CUDA masks are treated as
  unavailable telemetry before partial NVML handle lookup. ROCm utilization
  resolves `ROCR_VISIBLE_DEVICES` as the base mask and one matching
  `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm
  SMI. If a mapping cannot be resolved, utilization is treated as unavailable.
- `interval` is the finite positive pause between keep-alive bursts inside the
  background thread, capped by the Python runtime wait limit.
- `vram_to_keep` accepts integer bytes or human-readable strings (`parse_size`
  handles it). Byte-equivalent values above 1 PiB are rejected.
- When omitted, direct `CudaGPUController`, `RocmGPUController`, and
  `MacMGPUController` constructors default `vram_to_keep` to the shared
  low-power public default, `1GiB`.
- Platform-specific keep workload iteration counts must be positive integers.
  CUDA uses `relu_iterations`; ROCm and Mac M use `iterations`. Rejecting
  non-integer and non-positive values prevents a keep session from starting
  with no useful keep-alive work.

## Start/stop manually

Need more control? Call `keep()` and `release()` yourself.

```python
ctrl = CudaGPUController(rank=1, interval=1.0, vram_to_keep=1_073_741_824)

ctrl.keep()
run_cpu_bound_stage()
ctrl.release()
```

The controller spins up a daemon thread. Repeated `keep()` calls are idempotent
and simply warn if the worker is already running. CUDA and ROCm `keep()` calls
return only after fatal backend startup setup succeeds, so startup failures such
as device-selection errors are raised before your guarded work begins.
Invalid direct `rank` values are rejected even earlier, during construction.

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
- Local constructor inputs (`gpu_ids`, `interval`, `busy_threshold`, and
  `vram_to_keep`) are validated before platform/backend discovery. Visible-count
  checks for explicit IDs still require device discovery.
- `vram_to_keep` defaults to the shared low-power public default, `1GiB`, when
  omitted. Pass an explicit smaller or larger value when your scheduler needs a
  different reservation signal.
- `busy_threshold` defaults to `25` and accepts `-1` or a percentage in
  `0..100`. Non-negative thresholds throttle the keep-alive loop when
  utilization spikes. When utilization telemetry is unavailable, non-negative
  thresholds sleep before allocating the keep tensor or running compute; use
  `busy_threshold=-1` only for explicit unconditional keepalive work.
- `release()` uses threads too, so all GPUs free up quickly. If a stop times out
  but the worker exits before a later release attempt, KeepGPU still clears the
  backend cache before forgetting the stale worker state.

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
