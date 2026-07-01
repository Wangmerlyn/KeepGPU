# How KeepGPU Works

At runtime, KeepGPU spins up one lightweight worker per GPU. Each worker keeps a
tensor allocated and runs a short backend-specific keepalive burst, then sleeps.
This convinces most schedulers that the GPU is still busy, without burning a
full training workload.

## Components

1. **CLI (Typer/Rich)** – Parses options, validates visible GPU ordinals, and configures the logger.
2. **`GlobalGPUController`** – Validates local constructor inputs before
   platform probes, detects the current platform (CUDA, ROCm, or Mac M series),
   validates selected visible ordinals against the current device count, and
   instantiates one single-GPU controller per selected device.
3. **`CudaGPUController`** / **`RocmGPUController`** / **`MacMGPUController`** –
   Platform-specific implementations for per-GPU keep-alive loops. Direct
   CUDA/ROCm controllers validate `rank` as a visible ordinal during
   construction, before creating device handles or starting workers.
4. **GPU monitor (NVML/ROCm/MPS)** – Wraps `nvidia-ml-py` (the `pynvml`
   module) for CUDA telemetry, system-provided ROCm SMI when `rocm_smi` is
   importable, and best-effort MPS memory counters on Mac M series.
   CUDA telemetry resolves `CUDA_VISIBLE_DEVICES` before querying NVML, accepts
   unique UUID prefixes, stops at `-1`, and treats malformed, duplicate, or
   ambiguous masks as unavailable telemetry.
   ROCm
   telemetry resolves `ROCR_VISIBLE_DEVICES` plus one matching
   `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm
   SMI. Utilization can be unavailable on some platforms and is reported as
   `null`; vendor readings outside finite `0..100` are treated as unavailable
   instead of idle.
5. **Utilities** – `parse_size` turns strings like `1GiB` or bare byte values into
   internal float32 tensor element counts, while `setup_logger` wires both console
   and file logging with optional colors.

```text
CLI args ──▶ GlobalGPUController ──▶ [backend controller rank=0]
                                      [backend controller rank=1]
                                      [...]
```

## Lifecycle

1. The CLI (or your Python code) instantiates `GlobalGPUController`; invalid
   local inputs fail before backend discovery.
2. During `keep()` / `__enter__`, `GlobalGPUController` starts each worker.
   If a later worker fails to start, already-started workers are released before
   the original start error is re-raised.
3. Each CUDA worker:
   - Has already validated its direct visible `rank` against the current CUDA
     device count before `torch.device`, `set_device`, telemetry, or allocation
     can target an invalid ordinal.
   - Starts a daemon thread and confirms fatal backend startup setup before
     `keep()` reports success.
   - Performs intervalled lightweight elementwise batches after startup.
   - Calls `_monitor_utilization` (by way of NVML) to detect real activity
     before allocating the keep tensor.
   - Allocates a tensor sized by way of `vram_to_keep` only when backoff allows
     work.
     The monitor receives the CUDA visible rank and resolves
     `CUDA_VISIBLE_DEVICES` numeric tokens, unique UUID prefixes, or full UUID
     tokens before querying NVML, so telemetry follows the same device the
     worker keeps. CUDA parsing stops at `-1`; masks such as `0,2,-1,1` expose
     only the valid prefix before `-1`. Empty tokens, duplicate/equivalent
     masks, ambiguous UUID prefixes, and out-of-range indexes are treated as
     unavailable telemetry, so eco-safe backoff applies instead of aliasing or
     guessing a physical GPU.
4. Each ROCm worker follows the same visible-rank contract. The controller keeps
   the visible rank selected by the user, while ROCm SMI telemetry is queried by
   the resolved physical SMI index when the environment masks are numeric,
   unique, and unambiguous. If the mapping cannot be resolved, telemetry is
   unavailable for that cycle.
5. If utilization exceeds `busy_threshold`, or if utilization is unavailable
   while `busy_threshold` is non-negative, the worker just sleeps for one more
   `interval` before allocating or running ops. Unavailable includes malformed,
   boolean, non-finite, or out-of-range vendor utilization readings. Otherwise
   it allocates the keep tensor when needed and runs a batch. Public intervals
   must be finite positive seconds within the Python runtime wait limit. Public
   VRAM byte-equivalent inputs are capped at 1 PiB before conversion to tensor
   elements. Public defaults use `busy_threshold=25`. Valid thresholds are `-1`
   or `0..100`; `busy_threshold=-1` is the explicit unconditional mode.
6. When you call `release()` (or exit the context), every worker sets a stop
   event, joins the thread, and clears the device cache. Release attempts every
   worker and then raises a summary if any worker failed to stop. If a timed-out
   worker exits before a later release attempt, that later release still clears
   the backend cache before dropping stale thread state.

## Why lightweight elementwise batches?

Elementwise keep-alive batches:

- Allocate continuous VRAM quickly, which is what schedulers monitor.
- Exercise compute units enough to show non-zero utilization spikes.
- Are deterministic and easy to tune with interval and positive integer
  iteration settings, trading power draw for stronger "busy" signals.
  Non-integer and non-positive iteration counts are rejected so a keep session
  cannot crash later or silently do no useful keep-alive work.

## Threading & responsiveness

- The keep-alive loop runs on daemon threads so the main process can exit fast.
- `GlobalGPUController.release()` stops workers concurrently by way of threads, keeping
  shutdown time bounded even with many GPUs.
- Service stop-all releases independent sessions concurrently after taking its
  session snapshot, while aggregating results in deterministic snapshot order.
- Service session state is intentionally conservative: `status` shows reserved
  jobs as `state="starting"` while controller startup is in progress, and
  `stop_keep` removes a session only after release succeeds. Stop requests wait
  for starting jobs within a bounded budget; if that wait times out, the
  response uses `timed_out`, status shows the remembered cancellation as
  `state="stopping"` with the timeout message, and a later successful startup
  is released in the background. Timed-out releases stay visible as
  `state="stopping"` until the background release finishes; failed releases
  stay visible as `state="stop_failed"` with `last_error`.
- After startup, terminal worker runtime or allocation failures are surfaced as
  `state="runtime_failed"` with `last_error`. The retained session remains
  visible and can still be stopped. Busy-GPU or unavailable-telemetry deferral is
  normal backoff behavior and does not change an active session to
  `runtime_failed`.
- Fatal backend startup errors are reported before `keep()` returns. Recoverable
  later runtime errors are logged, and recoverable allocation failures retry
  after clearing the device cache.

## Platform detection

`get_platform()` inspects the system and enables the CUDA, ROCm, or Mac M series
(MPS) path. Detection order: CUDA → ROCm → Mac M → CPU fallback.
Vendor detection probes initialize and then shut down their telemetry libraries
immediately, so detection does not leave NVML or ROCm SMI handles open. A PyTorch
build with a truthy `torch.version.hip` is treated as ROCm before any NVML-based
CUDA fallback, even if NVML is available on the host.

Runtime telemetry has its own recovery path because GPU listing can run while a
keeper is active. If an independent listing probe shuts down NVML or ROCm SMI,
CUDA and ROCm utilization monitors reinitialize once before reporting telemetry
as unavailable, preserving eco-safe backoff without wedging a keeper on stale
library state.

GPU listing follows the same precedence. On HIP/ROCm torch builds, `list_gpus`
prefers ROCm records and ROCm SMI metadata instead of returning NVML CUDA records
first on mixed hosts. If ROCm SMI is unavailable, listing falls back to
torch's HIP-backed device records rather than NVML CUDA records. ROCm records
are emitted only for visible ordinals that `torch.cuda.set_device()` can select,
so list output does not advertise GPU IDs that controller startup cannot use.
Memory probing remains best-effort after selection succeeds; nullable ROCm
memory fields mean unavailable telemetry, not a startability failure.
On non-HIP CUDA builds, NVML records are returned only when Torch CUDA reports a
matching positive visible-device count and each visible ordinal can be selected
with `torch.cuda.set_device()`. If that NVML/Torch trust check fails, listing
falls back to Torch CUDA records and does not probe ROCm SMI on non-HIP builds,
so CUDA devices are not misreported as ROCm inventory.
