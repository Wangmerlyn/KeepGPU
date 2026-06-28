# How KeepGPU Works

At runtime, KeepGPU spins up one lightweight worker per GPU. Each worker keeps a
tensor allocated and runs a burst of CUDA ops, then sleeps. This convinces most
schedulers that the GPU is still busy, without burning a full training workload.

## Components

1. **CLI (Typer/Rich)** – Parses options, validates visible GPU ordinals, and configures the logger.
2. **`GlobalGPUController`** – Detects the current platform (CUDA, ROCm,
   or Mac M series), validates selected visible ordinals against the current
   device count, and instantiates one single-GPU controller per selected device.
3. **`CudaGPUController`** / **`RocmGPUController`** / **`MacMGPUController`** –
   Platform-specific implementations for per-GPU keep-alive loops.
4. **GPU monitor (NVML/ROCm/MPS)** – Wraps `nvidia-ml-py` (the `pynvml`
   module) for CUDA telemetry, optionally `rocm-smi` when installed by way of
   the `rocm` extra, and best-effort MPS memory counters on Mac M series.
   CUDA telemetry resolves `CUDA_VISIBLE_DEVICES` before querying NVML and
   treats duplicate or ambiguous masks as unavailable telemetry. ROCm
   telemetry resolves `ROCR_VISIBLE_DEVICES` plus one matching
   `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm
   SMI. Utilization can be unavailable on some platforms and is reported as
   `null`.
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
   - Starts a daemon thread that performs intervalled lightweight elementwise batches.
   - Calls `_monitor_utilization` (by way of NVML) to detect real activity
     before allocating the keep tensor.
   - Allocates a tensor sized by way of `vram_to_keep` only when backoff allows
     work.
     The monitor receives the CUDA visible rank and resolves
     `CUDA_VISIBLE_DEVICES` numeric or UUID tokens before querying NVML, so
     telemetry follows the same device the worker keeps. Duplicate masks are
     treated as unavailable telemetry instead of aliasing two visible ranks to
     one physical GPU.
4. Each ROCm worker follows the same visible-rank contract. The controller keeps
   the visible rank selected by the user, while ROCm SMI telemetry is queried by
   the resolved physical SMI index when the environment masks are numeric,
   unique, and unambiguous. If the mapping cannot be resolved, telemetry is
   unavailable for that cycle.
5. If utilization exceeds `busy_threshold`, or if utilization is unavailable
   while `busy_threshold` is non-negative, the worker just sleeps for one more
   `interval` before allocating or running ops. Otherwise it allocates the keep
   tensor when needed and runs a batch. Public defaults use `busy_threshold=25`.
   Valid thresholds are `-1` or `0..100`; `busy_threshold=-1` is the explicit
   unconditional mode.
6. When you call `release()` (or exit the context), every worker sets a stop
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
- Service stop-all releases independent sessions concurrently after taking its
  session snapshot, while aggregating results in deterministic snapshot order.
- Service session state is intentionally conservative: `status` shows reserved
  jobs as `state="starting"` while controller startup is in progress, and
  `stop_keep` removes a session only after release succeeds. Timed-out sessions
  stay visible as `state="stopping"` until the background release finishes;
  failed releases stay visible as `state="stop_failed"` with `last_error`.
- Errors inside a worker are logged but do not bring the whole process down;
  the loop retries after clearing the CUDA cache.

## Platform detection

`get_platform()` inspects the system and enables the CUDA, ROCm, or Mac M series
(MPS) path. Detection order: CUDA → ROCm → Mac M → CPU fallback.
Vendor detection probes initialize and then shut down their telemetry libraries
immediately, so detection does not leave NVML or ROCm SMI handles open.
