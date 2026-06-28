# Agent Guidelines

This file defines how coding agents should work in this repository.

## 1) General (Reusable)

### Language

- Default to English for all comments, docs, user-facing copy, and logs.
- Use non-English text only when required for i18n/localization.

### Workflow (Branches + PRs)

- Always branch from the latest `main` when starting a new feature or bug fix.
- Use project-local worktrees under `.worktrees/` for parallel agent work.
  - Example: `git worktree add .worktrees/codex/<topic> -b codex/<topic> origin/main`
- Implement work on a new branch, validate changes, then open a PR to `main` for review.
- Keep commits small and focused; avoid mixing unrelated changes.

### Planning (When Work Is Non-trivial)

- If the task is complex or ambiguous, propose a short plan and confirm it with the user before large changes.
- Before starting complex work, capture background, goal, solution, and todo items in a Markdown plan under `docs/plans/`.
- Implementations must follow the plan and its todo items; update the plan document when tasks or scope change.
- If requirements are unclear during planning, ask the user early and proceed only after confirmation.
- If a plan-related subagent exists, prefer calling it to draft/refine the plan.

### Code Review

- If a review-related subagent exists, call it when the code is ready for review.
- Address review findings as appropriate until no must-fix issues remain.
- Run local subagent code review before merging any PR; squash merge only after all review comments are resolved.

### Documentation Updates

- When adding user-visible features, update the user documentation with usage guidance alongside the code changes.
- For complex features, bug investigations, or refactorings that require detailed documentation (for example, plans, testing guides, summaries), create a dedicated subfolder under `docs/` with a descriptive name (for example, `docs/opencode-poll-loop-refactor/`). Place all related documentation files in that subfolder.
- Avoid placing new project-specific documentation in the root directory; keep only canonical top-level docs (for example, `AGENTS.md`, `README.md`) at root.

### Quality Bar

- Prefer root-cause fixes over defensive patches.
- Keep it "Linus" simple - concise, readable, and robust; avoid bloat/over-engineering.
- Run the smallest relevant checks first (unit tests, targeted scripts), then broader checks when needed.
- Add tests when there is an existing test pattern; do not introduce a brand-new testing framework unless requested.

### Git Hygiene & Security

- Commit messages: use `type(scope): summary`.
- PR titles: format `[modules] type: description` (modules comma-separated, single type).
- Never commit secrets (tokens, credentials files).
- Avoid destructive git operations unless explicitly requested (for example, `reset --hard`, force-push).

## 2) Project-Specific (KeepGPU)

### Product Surface (Keep These in Sync)

- This repository has three first-class interfaces:
  - CLI: `keep-gpu` (`src/keep_gpu/cli.py`)
  - Python API/controllers (`src/keep_gpu/single_gpu_controller/`, `src/keep_gpu/global_gpu_controller/`)
  - MCP server: `keep-gpu-mcp-server` (`src/keep_gpu/mcp/server.py`)
- If behavior changes in one interface, update related docs/tests for that interface and check if parity is required in the others.
- Keep the MCP server compatible with standard MCP lifecycle/tool methods
  (`initialize`, `notifications/initialized`, `tools/list`, `tools/call`) while
  preserving legacy direct JSON-RPC method calls for local scripts.
- JSON-RPC handlers must reject explicit request versions other than `"2.0"`
  with `-32600 Invalid Request` for request messages while preserving
  omitted-version legacy/internal calls and silent id-less notifications.
- HTTP JSON-RPC endpoints (`/` and `/rpc`) must return JSON-RPC envelopes for
  protocol parse errors, including `jsonrpc`, `id`, and numeric `error.code`;
  REST routes keep REST-shaped structured JSON errors.
- HTTP body readers must validate `Content-Length` as a non-negative integer
  before reading; invalid values must fail quickly with REST JSON 400 errors or
  JSON-RPC `-32700` parse-error envelopes.
- For stdio MCP, stdout must contain only JSON protocol messages; diagnostics
  and human logs belong on stderr.

### Architecture Boundaries

- Keep platform detection and environment probing centralized in `src/keep_gpu/utilities/platform_manager.py`.
- Keep GPU telemetry helpers in `src/keep_gpu/utilities/gpu_info.py` and related utility modules.
- Keep public session input validation centralized in `src/keep_gpu/utilities/session_config.py`; CLI, Python, REST, and JSON-RPC entry points must share the same contract.
- JSON-RPC user parameter validation errors and unknown direct-method params must return `-32602 Invalid params`; reserve `-32603 Internal error` for unexpected server failures.
- Expected controller startup unavailability, such as no usable visible GPUs or
  unsupported platforms, must surface as explicit startup-unavailable errors
  (direct JSON-RPC `-32000`, REST `503`, MCP tool `isError=true`) while
  arbitrary unexpected startup/runtime failures remain internal errors.
- CLI service JSON commands (`status`, `stop`, `list-gpus`) must print structured JSON objects that downstream tools can parse with one decode, including `{"error": "..."}` objects for service/runtime errors after CLI parsing succeeds.
- CLI service RPC clients must reject malformed JSON-RPC service envelopes (wrong `jsonrpc`, mismatched `id`, missing `result`, non-object direct-method `result`, or non-object `error`) instead of treating them as successful empty responses or leaking tracebacks.
- CLI commands that consume service-specific JSON-RPC results must validate
  required result fields before rendering user-facing output; malformed
  method-specific payloads should become clean `ServiceResponseError` messages,
  not `KeyError`, tracebacks, or partial success text.
- CLI service endpoint inputs (`--host`, `--port`) must be validated locally
  before service RPC, daemon auto-start, stop-all fallback, or daemon ownership
  operations. JSON-output commands must return structured `{"error": "..."}`
  objects for invalid endpoints, not tracebacks.
- MCP executable HTTP endpoint inputs (`--host`, `--port`) must be validated
  before socket bind, matching the CLI service endpoint contract.
- `keep-gpu start` must validate local inputs such as `--vram`, `--job-id`, `--interval`, `--busy-threshold`, `--gpu-ids`, `--host`, and `--port` before auto-starting the service daemon or making RPC calls.
- Blocking-mode root CLI options (`--gpu-ids`, `--vram`,
  `--busy-threshold`/`--util-threshold`, hidden `--threshold`, and
  `--interval`) must be rejected when explicitly supplied before a service
  subcommand; service options belong after the subcommand, for example
  `keep-gpu start --gpu-ids 0`.
- `keep-gpu status --job-id` and `keep-gpu stop --job-id` must validate
  explicit custom IDs locally before service RPC, stop-all fallback, or daemon
  side effects. Only omitting the option means all-session status or no stop
  target was chosen.
- For CLI `--gpu-ids`, only omission means all visible GPUs; explicit empty or
  whitespace-only values are invalid and must not silently expand to all GPUs.
- Destructive or broad stop surfaces must reject ambiguous inputs such as `--job-id` with `--all` before any RPC, stop-all fallback, or daemon stop side effect.
- REST session creation bodies must be JSON objects; reject arrays/scalars before field validation or session state changes.
- REST session creation must validate cheap local fields (`vram`, `interval`,
  `busy_threshold`, `job_id`, duplicate custom `job_id`, and `gpu_ids` shape)
  before telemetry/list_gpus probing. Valid explicit `gpu_ids` are still checked
  against listed visible IDs before startup.
- Supported REST API routes/methods must return structured JSON error objects
  for validation, unknown-endpoint, and unexpected runtime failures; do not let
  handler exceptions drop the HTTP connection.
- The dashboard request wrapper must prefer structured REST `error.message`
  strings over raw JSON bodies so users see actionable start/refresh/release
  failures instead of protocol payloads.
- Public `interval` values must be finite positive seconds, including fractional seconds, capped by the Python runtime wait limit; reject `NaN`, `Infinity`, and oversized values before creating or mutating session state so keep loops cannot spin, crash, or wedge.
- Keep human VRAM parsing centralized in `src/keep_gpu/utilities/humanized_input.py`; public integer values and digit-only strings mean bytes, public VRAM byte-equivalent values must be no more than 1 PiB, and controllers may convert valid inputs to internal tensor element counts.
- Keep omitted public VRAM defaults aligned and low-power: CLI, Python global controller, Python single-GPU controllers, service, REST, JSON-RPC, and MCP should default to `1GiB` per GPU unless a user explicitly asks for a different reservation.
- `GlobalGPUController` must validate local constructor inputs (`gpu_ids`,
  `interval`, `busy_threshold`, `vram_to_keep`) before calling
  `get_platform()` or other hardware/backend probes. Visible-count checks for
  explicit IDs still happen after platform/device discovery.
- Hardware probes must clean up vendor libraries after detection (for example, NVML shutdown and ROCm SMI shutdown after init).
- ROCm/HIP PyTorch builds take precedence over NVML-based CUDA fallback: if `torch.version.hip` is truthy, `_check_cuda()` must not classify the runtime as CUDA or probe NVML.
- Keep lifecycle state truthful: a reserved starting session must be visible in status as `state="starting"`; a session is removed only after release succeeds; timed-out or failed stops must stay visible with state and error details.
- Release paths must be idempotent after timeout: when a previously stopping
  worker has since died, perform backend cache cleanup and clear stale thread
  and stop-event state instead of returning early as not running.
- Runtime/allocation failures reported by a started worker must be retained in
  status as `state="runtime_failed"` with `last_error`; the session remains
  visible and stoppable. Busy or unavailable telemetry backoff is normal
  deferred allocation behavior, not a runtime failure.
- Controller runtime-health hooks used by service status must stay non-blocking
  and read-only because status refresh runs under the session lock.
- Single-GPU `keep()` must not report success until fatal backend startup setup
  has succeeded. CUDA/ROCm worker startup failures such as `set_device` errors
  must propagate synchronously so services cannot register false active sessions.
- Keep service daemon ownership safe: no stop, force-stop, or fallback path may signal a PID unless the auto-start ownership record verifies the running process.
- Non-force `keep-gpu service-stop` must require a reachable service and a successful status/RPC check before signaling; use `--force` for unresponsive auto-started daemons.
- Treat custom `job_id` values as reserved from the moment startup begins; duplicate starts must fail before another controller can begin keep-alive work.
- Keep custom `job_id` validation centralized in `session_config.py`: only `None` means omitted/all-sessions, and custom IDs must be non-empty URL-path-safe strings before any session state changes.
- Stop requests must not miss starting sessions; wait for startup to settle before returning `not found` or taking a stop-all snapshot.
- Stop-all may release independent sessions concurrently, but must not duplicate release work for `stopping` sessions and must keep deterministic additive result fields.
- Keep utilization backoff eco-safe: valid `busy_threshold` values are `-1` or `0..100`; public defaults must use the shared `DEFAULT_BUSY_THRESHOLD` (`25`), and when telemetry is unavailable with `busy_threshold >= 0`, controllers should sleep instead of allocating keep tensors or running keepalive compute. Only `busy_threshold=-1` is the explicit unconditional mode.
- Dashboard telemetry must display unavailable utilization as unknown/`n/a`, not
  as `0%` idle; aggregate utilization summaries must ignore unavailable readings
  and show `n/a` when no finite readings exist, and per-GPU utilization bars
  must not render an idle fill for unavailable telemetry.
- Dashboard labels, CLI logs, and other user-facing sentinel renderings must
  show semantic meaning instead of raw numeric shapes; for example,
  `busy_threshold=-1` is unconditional keepalive mode, not `-1%` utilization.
- Single-GPU keep workload iteration counts must be positive integers (`relu_iterations` for CUDA, `iterations` for ROCm/Mac M); reject invalid values before keep loops so no public path can create a silent no-op keeper or late background thread crash.
- ROCm optional allocation retry counts must be `None` or positive plain
  integers; reject invalid values before worker startup so background retry
  loops cannot crash with type errors after `keep()` returns.
- Direct CUDA/ROCm single-GPU `rank` values must be plain integer visible
  device ordinals validated against the current visible device count during
  construction, before `torch.device`, backend `set_device`, telemetry, or keep
  worker startup can target an invalid ordinal.
- Treat public `gpu_ids` as visible device ordinals after user-supplied CUDA or ROCm visibility filtering. Do not rewrite visibility masks inside KeepGPU command paths; reject explicit CUDA/ROCm ordinals outside the current visible device count before starting keep workers.
- Keep CUDA telemetry aligned with visible CUDA ordinals: `get_gpu_utilization(index)` receives the visible rank used by `CudaGPUController`, and `gpu_monitor.py` resolves `CUDA_VISIBLE_DEVICES` numeric/UUID tokens to the correct NVML handle. If that mapping is malformed, duplicate/equivalent, ambiguous, or unsupported, return `None` without partial NVML handle queries so eco-safe backoff applies instead of falling back to a possibly wrong physical index.
- Keep ROCm telemetry aligned with visible ROCm ordinals: resolve `ROCR_VISIBLE_DEVICES` as the base mask and one matching `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` overlay before querying ROCm SMI. If the mapping is malformed, conflicting, unsupported, or out of range, return unavailable utilization rather than querying a guessed SMI index.
- Keep GPU listing IDs aligned with start APIs: `list_gpus`/`/api/gpus` must expose `id` as the visible ordinal users can pass as `gpu_ids`; physical/vendor identifiers belong in explicit metadata fields such as `physical_id` and must not be accepted implicitly as selection IDs.
- Keep GPU listing platform precedence aligned with controller platform
  detection: HIP/ROCm torch builds must prefer ROCm listing over NVML CUDA
  listing, and must not fall back to NVML CUDA records when torch's active
  runtime is HIP.
- Global controller startup must fail clearly when GPU selection resolves to zero or duplicate devices; do not create silent no-op or duplicate-worker keep sessions.
- Avoid scattering platform-specific branching across unrelated modules; prefer one clear decision path then platform-specific controller classes.
- Preserve simple controller flow: global controller orchestrates per-GPU controllers; single-GPU controllers handle device-level keep/release loops.

### Platform and Dependency Rules

- CUDA telemetry should rely on `nvidia-ml-py` (imported as `pynvml` module), not the deprecated standalone `pynvml` package.
- ROCm support is optional and should remain guarded (`rocm-smi` in extras). Imports must fail gracefully on non-ROCm machines.
- Apple Silicon/MPS telemetry is best-effort; return a guarded MACM record with nullable memory/utilization fields rather than hiding the device.
- CI runners generally do not have GPUs. GPU-dependent logic must have safe fallbacks and tests must avoid hard-failing in no-GPU environments.

### Testing Expectations in This Repository

- Before pushing, run targeted tests relevant to changed modules first, then broader checks.
- Common targeted commands:
  - `pytest tests/cuda_controller tests/global_controller tests/utilities/test_platform_manager.py`
  - `pytest tests -k threshold`
  - `pytest tests/mcp tests/utilities/test_gpu_info.py`
  - `pytest tests/rocm_controller/test_rocm_utilization.py tests/utilities/test_gpu_info.py`
  - `pytest --run-rocm tests/rocm_controller` (only on ROCm-capable machines)
- Respect existing pytest markers:
  - `rocm` for ROCm-only tests
  - `large_memory` for opt-in heavy VRAM tests
- Run `pre-commit run --all-files` before final push.

### Documentation and Build Expectations

- Update docs when changing CLI flags, controller behavior, platform support, or MCP methods.
- Keep README and `docs/` guidance aligned for install and usage flows.
- Validate docs with:
  - `mkdocs build` for static build verification
  - `mkdocs serve` for local preview when editing rendered content
- Do not commit generated `site/` output unless explicitly requested.

### Scope and Release Hygiene

- Keep commits narrow and module-focused (especially when touching controllers and platform utilities).
- Do not bump versions, move tags, or alter release metadata unless explicitly requested.
- When fixing review comments, prefer minimal diffs that address the exact finding and keep existing behavior stable.
