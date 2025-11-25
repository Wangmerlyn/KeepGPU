# How KeepGPU Works

At runtime, KeepGPU spins up one lightweight worker per GPU. Each worker keeps a
tensor allocated and runs a burst of CUDA ops, then sleeps. This convinces most
schedulers that the GPU is still busy, without burning a full training workload.

## Components

1. **CLI (Typer/Rich)** – Parses options, validates GPU IDs, and configures the logger.
2. **`GlobalGPUController`** – Detects the current platform (CUDA today) and
   instantiates one single-GPU controller per selected device.
3. **`CudaGPUController`** – Owns the background thread, VRAM allocation, and small
   matmul loops that tick every `interval` seconds.
4. **GPU monitor (NVML/ROCm)** – Wraps `nvidia-ml-py` (the `pynvml` module) for CUDA
   telemetry and optionally `rocm-smi` when installed via the `rocm` extra.
5. **Utilities** – `parse_size` turns strings like `1GiB` into bytes, while
   `setup_logger` wires both console and file logging with optional colors.

```text
CLI args ──▶ GlobalGPUController ──▶ [CudaGPUController rank=0]
                                      [CudaGPUController rank=1]
                                      [...]
```

## Lifecycle

1. The CLI (or your Python code) instantiates `GlobalGPUController`.
2. During `keep()` / `__enter__`, each Cuda worker:
   - Allocates a tensor sized by way of `vram_to_keep`.
   - Starts a daemon thread that performs `matmul_iterations` fused activations.
   - Calls `_monitor_utilization` (by way of NVML) to detect real activity.
3. If utilization exceeds `busy_threshold`, the worker just sleeps for one more
   `interval`. Otherwise it runs a new batch of ops.
4. When you call `release()` (or exit the context), every worker sets a stop
   event, joins the thread, and clears the CUDA cache.

## Why matmuls?

Matrix multiplies:

- Allocate continuous VRAM quickly, which is what schedulers monitor.
- Exercise compute units enough to show non-zero utilization spikes.
- Are deterministic and easy to tune—adjust `matmul_iterations` to trade power
  draw for stronger “busy” signals.

## Threading & responsiveness

- The keep-alive loop runs on daemon threads so the main process can exit fast.
- `GlobalGPUController.release()` stops workers concurrently by way of threads, keeping
  shutdown time bounded even with many GPUs.
- Errors inside a worker are logged but do not bring the whole process down;
  the loop retries after clearing the CUDA cache.

## Platform detection

`get_platform()` inspects the system and currently only enables the CUDA path.
If you plan to contribute ROCm or CPU fallbacks, use this hook to branch into
new controller implementations without changing the CLI.
