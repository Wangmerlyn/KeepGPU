# How KeepGPU Works

At runtime, KeepGPU spins up one lightweight worker per GPU. Each worker keeps a
tensor allocated and runs a burst of CUDA ops, then sleeps. This convinces most
schedulers that the GPU is still busy, without burning a full training workload.

## Components

1. **CLI (Typer/Rich)** – Parses options, validates GPU IDs, and configures the logger.
2. **`GlobalGPUController`** – Detects the current platform (CUDA, ROCm,
   or Mac M series) and instantiates one single-GPU controller per selected device.
3. **`CudaGPUController`** / **`RocmGPUController`** / **`MacMGPUController`** –
   Platform-specific implementations for per-GPU keep-alive loops.
4. **GPU monitor (NVML/ROCm/MPS)** – Wraps `nvidia-ml-py` (the `pynvml`
   module) for CUDA telemetry, optionally `rocm-smi` when installed by way of
   the `rocm` extra, and best-effort MPS memory counters on Mac M series.
   Utilization can be unavailable on some platforms and is reported as `null`.
5. **Utilities** – `parse_size` turns strings like `1GiB` or bare byte values into
   internal float32 tensor element counts, while `setup_logger` wires both console
   and file logging with optional colors.

```text
CLI args ──▶ GlobalGPUController ──▶ [CudaGPUController rank=0]
                                      [CudaGPUController rank=1]
                                      [...]
```

## Lifecycle

1. The CLI (or your Python code) instantiates `GlobalGPUController`.
2. During `keep()` / `__enter__`, `GlobalGPUController` starts each worker.
   If a later worker fails to start, already-started workers are released before
   the original start error is re-raised.
3. Each CUDA worker:
   - Allocates a tensor sized by way of `vram_to_keep`.
   - Starts a daemon thread that performs intervalled lightweight elementwise batches.
   - Calls `_monitor_utilization` (by way of NVML) to detect real activity.
4. If utilization exceeds `busy_threshold`, the worker just sleeps for one more
   `interval`. Otherwise it runs a new batch of ops.
5. When you call `release()` (or exit the context), every worker sets a stop
   event, joins the thread, and clears the device cache. Release attempts every
   worker and then raises a summary if any worker failed to stop.

## Why lightweight elementwise batches?

Elementwise keep-alive batches:

- Allocate continuous VRAM quickly, which is what schedulers monitor.
- Exercise compute units enough to show non-zero utilization spikes.
- Are deterministic and easy to tune with interval and iteration settings, trading
  power draw for stronger “busy” signals.

## Threading & responsiveness

- The keep-alive loop runs on daemon threads so the main process can exit fast.
- `GlobalGPUController.release()` stops workers concurrently by way of threads, keeping
  shutdown time bounded even with many GPUs.
- Service session state is intentionally conservative: `stop_keep` removes a
  session only after release succeeds. Timed-out sessions stay visible as
  `state="stopping"` until the background release finishes; failed releases stay
  visible as `state="stop_failed"` with `last_error`.
- Errors inside a worker are logged but do not bring the whole process down;
  the loop retries after clearing the CUDA cache.

## Platform detection

`get_platform()` inspects the system and enables the CUDA, ROCm, or Mac M series
(MPS) path. Detection order: CUDA → ROCm → Mac M → CPU fallback.
Vendor detection probes initialize and then shut down their telemetry libraries
immediately, so detection does not leave NVML or ROCm SMI handles open.
