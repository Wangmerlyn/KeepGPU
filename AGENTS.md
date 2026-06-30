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
- `docs/plans/` and `docs/skills/` are internal source artifacts and are
  excluded from the public MkDocs site.
- If requirements are unclear during planning, ask the user early and proceed only after confirmation.
- If a plan-related subagent exists, prefer calling it to draft/refine the plan.

### Code Review

- If a review-related subagent exists, call it when the code is ready for review.
- Address review findings as appropriate until no must-fix issues remain.
- Run local subagent code review before merging any PR; squash merge only after all review comments are resolved.

### Documentation Updates

- When adding user-visible features, update the user documentation with usage guidance alongside the code changes.
- For complex features, bug investigations, or refactorings that require detailed documentation (for example, plans, testing guides, summaries), create a dedicated subfolder under `docs/` with a descriptive name (for example, `docs/opencode-poll-loop-refactor/`). Place all related documentation files in that subfolder.
- Keep `README.md` as a concise front door. Put detailed CLI/API/MCP/platform
  contracts in `docs/` pages and link to them from README instead of repeating
  reference material.
- Avoid placing new project-specific documentation in the root directory; keep only canonical top-level docs (for example, `AGENTS.md`, `README.md`) at root.

### Quality Bar

- Prefer root-cause fixes over defensive patches.
- Keep it "Linus" simple - concise, readable, and robust; avoid bloat/over-engineering.
- Run the smallest relevant checks first (unit tests, targeted scripts), then broader checks when needed.
- Add tests when there is an existing test pattern; do not introduce a brand-new testing framework unless requested.
- Build metadata should list directly used third-party distributions; do not
  rely on transitive dependencies, and do not add Python stdlib modules such as
  `argparse` as build/runtime dependencies.
- Pre-commit CI should install only the `pre-commit` runner; hooks provision
  their own tool environments, so do not install KeepGPU runtime dependencies
  just to lint.
- Keep Python test CI dependency installs explicit. Do not reintroduce a root
  `requirements.txt` fallback; runtime and test dependencies belong in
  `pyproject.toml`, while docs dependencies belong in `docs/requirements.txt`.
- Source distributions should not ship the test suite by default; package data
  should enumerate required runtime assets such as the MCP dashboard files.
- Keep license metadata as a plain SPDX string and keep the advertised Python
  support floor aligned with build-backend requirements.
- Cosmetic logging helpers must stay optional; stdlib logging fallback is part
  of the supported runtime path.
- Keep `project.requires-python` aligned with the documented Python support
  floor and py39-targeted tooling.
- Keep `project.urls` entries live and aligned with current repository
  locations; avoid stale branch names or missing files.
- Remove placeholder tests that assert no behavior instead of preserving
  count-only coverage.
- Keep metadata tests self-contained for simple checks; avoid importing parser
  libraries that are only available through transitive test dependencies.
- Python files that use PEP 604 annotations such as `X | Y` must include
  `from __future__ import annotations` while the project tooling targets
  Python 3.9.
- MkDocs/mkdocstrings must resolve API references from the checkout's `src/`
  tree so docs-only builds work with `docs/requirements.txt` alone.
- `docs/requirements.txt` must list docs tools invoked directly by CI or
  contributors, including `mkdocs`; do not rely on theme/plugin transitive
  dependencies for documented commands.

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
- JSON-RPC error envelopes must use `id: null` when the request id is missing or
  has an invalid response-id type, and must echo only valid non-boolean string or
  integer ids.
- HTTP JSON-RPC endpoints (`/` and `/rpc`) must return JSON-RPC envelopes for
  protocol parse errors, including `jsonrpc`, `id`, and numeric `error.code`;
  REST routes keep REST-shaped structured JSON errors.
- HTTP body readers must validate `Content-Length` as a non-negative integer
  before reading; invalid values must fail quickly with REST JSON 400 errors or
  JSON-RPC `-32700` parse-error envelopes.
- API/RPC unsupported HTTP methods must not fall back to
  `BaseHTTPRequestHandler` HTML errors; known API/RPC routes return structured
  JSON `405 Method Not Allowed` responses with `Allow`, and exact `/api` plus
  unknown `/api/*` routes return structured JSON `404 Unknown endpoint`
  responses.
- Implemented HTTP verb handlers (`do_POST`, `do_DELETE`, and future siblings)
  must call the shared unsupported-method helper before local unknown-endpoint
  404 branches, so known paths never regress to 404 or body-parse errors solely
  because the wrong implemented verb was used.
- `/rpc` is an exact POST-only JSON-RPC endpoint; `GET /rpc` must return
  structured JSON `405 Method Not Allowed` with `Allow: POST`, while
  noncanonical spellings such as `/rpc/`, `/rpc;...`, and `/rpc?...` return
  structured JSON `404 Unknown endpoint` without JSON-RPC dispatch or
  dashboard/static fallback.
- Missing dashboard asset URLs, including `/assets/*` and extension-bearing
  static paths, must return structured JSON `404` errors instead of the
  dashboard HTML shell.
- For stdio MCP, stdout must contain only JSON protocol messages; diagnostics
  and human logs belong on stderr.

### Architecture Boundaries

- Keep platform detection and environment probing centralized in `src/keep_gpu/utilities/platform_manager.py`.
- Keep GPU telemetry helpers in `src/keep_gpu/utilities/gpu_info.py` and related utility modules.
- Keep public session input validation centralized in `src/keep_gpu/utilities/session_config.py`; CLI, Python, REST, and JSON-RPC entry points must share the same contract.
- JSON-RPC user parameter validation errors and unknown direct-method params must return `-32602 Invalid params`; reserve `-32603 Internal error` for unexpected server failures.
- Expected controller startup unavailability, such as no usable visible GPUs,
  failed CUDA/ROCm visible-device enumeration, or unsupported platforms, must
  surface as explicit startup-unavailable errors (direct JSON-RPC `-32000`,
  REST `503`, MCP tool `isError=true`) while arbitrary unexpected
  startup/runtime failures remain internal errors.
- CLI service JSON commands (`status`, `stop`, `list-gpus`) must print structured JSON objects that downstream tools can parse with one decode, including `{"error": "..."}` objects for service/runtime errors after CLI parsing succeeds.
- CLI service RPC clients must reject malformed JSON-RPC service envelopes (wrong `jsonrpc`, mismatched `id`, missing `result`, non-object direct-method `result`, or non-object `error`) instead of treating them as successful empty responses or leaking tracebacks.
- CLI commands that consume service-specific JSON-RPC results must validate
  required result fields before rendering user-facing output; malformed
  method-specific payloads should become clean `ServiceResponseError` messages,
  not `KeyError`, tracebacks, or partial success text.
- Keep `_rpc_call()` limited to generic JSON-RPC envelope validation; validate
  method-specific `status`, `stop_keep`, and `list_gpus` result payloads at CLI
  call sites before reading fields or triggering daemon stop side effects, while
  allowing extra result fields for forward compatibility.
- CLI method-specific result validation must include nested records consumed by
  downstream tools: `status.active_jobs` entries, disjoint `stop_keep` job-id
  outcome lists whose `errors` keys exactly match `failed`, and `list_gpus` GPU
  records with visible ordinal metadata. Known nested `status.params` fields
  must match the public session contract while extra fields remain
  forward-compatible.
- CLI service endpoint inputs (`--host`, `--port`) must be validated locally
  before service RPC, daemon auto-start, stop-all fallback, or daemon ownership
  operations. JSON-output commands must return structured `{"error": "..."}`
  objects for invalid endpoints, not tracebacks.
- Shared public endpoint validation belongs in
  `src/keep_gpu/utilities/endpoint_validation.py`; CLI and MCP entry points
  should only translate those `ValueError`s into their interface-specific error
  shape.
- JSON-output service commands (`status`, `stop`, `list-gpus`) must parse
  `--port` through command-level validation so non-integer values return the
  shared structured `{"error": "port must be an integer between 1 and 65535"}`
  object instead of Typer/Click usage text.
- CLI service daemon ownership checks must require known PID identity
  components. A PID record with `uid` or `start_time` missing or `null`, or a
  current process probe that cannot recover either value, is not ownership
  verified and must not be signaled as a managed daemon.
- MCP executable HTTP endpoint inputs (`--host`, `--port`) must be validated
  before socket bind, matching the CLI service endpoint contract.
- `keep-gpu start` must validate local inputs such as `--vram`, `--job-id`, `--interval`, `--busy-threshold`, `--gpu-ids`, `--host`, and `--port` before auto-starting the service daemon or making RPC calls.
- If `keep-gpu start` auto-starts a service daemon and the following
  `start_keep` RPC returns expected startup-unavailable JSON-RPC code
  `-32000` before creating a session, the CLI must best-effort stop that
  just-created daemon. Do not stop already-running daemons, malformed success
  payloads, or unrelated RPC failures.
- If service auto-start itself times out before the health check succeeds, the
  CLI must best-effort stop the just-started managed daemon and preserve the
  auto-start timeout error.
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
- Blocking CLI mode must defer omitted-GPU hardware enumeration to
  `GlobalGPUController`; the CLI should not perform an early
  `torch.cuda.device_count()` probe just to log the all-GPU count.
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
- Dashboard stop-result formatting must preserve non-empty backend `message`
  fields when no stop outcome lists are populated.
- Public `interval` values must be finite positive seconds, including fractional seconds, capped by the Python runtime wait limit; reject `NaN`, `Infinity`, and oversized values before creating or mutating session state so keep loops cannot spin, crash, or wedge.
- Keep human VRAM parsing centralized in `src/keep_gpu/utilities/humanized_input.py`; public integer values and digit-only strings mean bytes, public VRAM byte-equivalent values must be no more than 1 PiB, and controllers may convert valid inputs to internal tensor element counts.
- Keep omitted public VRAM defaults aligned and low-power: CLI, Python global controller, Python single-GPU controllers, service, REST, JSON-RPC, and MCP should default to `1GiB` per GPU unless a user explicitly asks for a different reservation.
- `GlobalGPUController` must validate local constructor inputs (`gpu_ids`,
  `interval`, `busy_threshold`, `vram_to_keep`) before calling
  `get_platform()` or other hardware/backend probes. Visible-count checks for
  explicit IDs still happen after platform/device discovery.
- Direct single-GPU controllers must validate cheap public constructor inputs
  before backend availability probes or device selection. For Mac M, validate
  `rank`, `busy_threshold`, and `iterations` before checking MPS availability.
  For CUDA and ROCm, reject non-integer `rank` values before calling
  `torch.cuda.device_count()`.
- Hardware probes must clean up vendor libraries after detection (for example, NVML shutdown and ROCm SMI shutdown after init).
- Runtime telemetry must tolerate those independent vendor-library probes. If a
  listing/detection path shuts down NVML or ROCm SMI after a keep loop has
  cached telemetry initialization, the runtime monitor should reinitialize once
  before reporting utilization as unavailable.
- ROCm/HIP PyTorch builds take precedence over NVML-based CUDA fallback: if `torch.version.hip` is truthy, `_check_cuda()` must not classify the runtime as CUDA or probe NVML.
- Keep lifecycle state truthful: a reserved starting session must be visible in status as `state="starting"`; a session is removed only after release succeeds; timed-out or failed stops must stay visible with state and error details.
- Status responses must be read-only snapshots. Returning active or starting
  session params must not expose the mutable objects stored in service state.
- Release paths must be idempotent after timeout: when a previously stopping
  worker has since died, perform backend cache cleanup and clear stale thread
  and stop-event state instead of returning early as not running.
- Runtime/allocation failures reported by a started worker must be retained in
  status as `state="runtime_failed"` with `last_error`; the session remains
  visible and stoppable. Busy or unavailable telemetry backoff is normal
  deferred allocation behavior, not a runtime failure.
- Controller runtime-health hooks used by service status must stay non-blocking
  and read-only because status refresh runs under the session lock.
- CUDA, ROCm, and MPS controllers must retain fatal post-start worker failures
  by way of `allocation_status()` so `GlobalGPUController.runtime_error()` can
  move service status to `runtime_failed`; normal busy/unknown telemetry backoff
  and out-of-memory allocation `RuntimeError` retries are not runtime failures.
  Non-OOM allocation or steady-state `RuntimeError` failures after startup must
  be retained as runtime failures.
- Single-GPU `keep()` must not report success until fatal backend startup setup
  has succeeded. CUDA/ROCm worker startup failures such as `set_device` errors
  must propagate synchronously so services cannot register false active sessions.
- Single-GPU `keep()` must reject a restart while a previous worker thread is
  still alive with its stop event already set; returning success in that state
  hides a stopping keeper as if a fresh keep succeeded.
- Keep service daemon ownership safe: no stop, force-stop, or fallback path may signal a PID unless the auto-start ownership record verifies the running process.
- Non-force `keep-gpu service-stop` must require a reachable service, successful status/RPC checks, and a clean `stop_keep` result with no timed-out or failed sessions before signaling; use `--force` for unresponsive auto-started daemons.
- Treat custom `job_id` values as reserved from the moment startup begins; duplicate starts must fail before another controller can begin keep-alive work.
- Keep custom `job_id` validation centralized in `session_config.py`: only `None` means omitted/all-sessions, and custom IDs must be non-empty URL-path-safe strings before any session state changes.
- Stop requests must not miss starting sessions. Targeted stops wait for the
  matching startup before returning `not found`; stop-all records its initial
  active/starting boundary first, waits only for starts in that boundary, and
  never includes later starts. After waiting, targeted stops must release only
  the session observed before the wait or the settled session matching the
  captured starting params, not a later same-`job_id` replacement.
- Stop requests waiting on starting sessions must be bounded. If startup does
  not settle within the stop wait budget, return the normal additive timeout
  payload, report the remembered cancellation in status as `state="stopping"`
  with a timeout `last_error`, and release a later successful startup in the
  background instead of exposing it as a surprise active keeper.
- Stop-all may release independent sessions concurrently, but must not duplicate release work for `stopping` sessions and must keep deterministic additive result fields.
- Keep utilization backoff eco-safe: valid `busy_threshold` values are `-1` or `0..100`; public defaults must use the shared `DEFAULT_BUSY_THRESHOLD` (`25`), and when telemetry is unavailable with `busy_threshold >= 0`, controllers should sleep instead of allocating keep tensors or running keepalive compute. Only `busy_threshold=-1` is the explicit unconditional mode.
- Dashboard telemetry must display unavailable utilization as unknown/`n/a`, not
  as `0%` idle; aggregate utilization summaries must ignore unavailable readings
  and show `n/a` when no finite readings exist, and per-GPU utilization bars
  must not render an idle fill for unavailable telemetry.
- Dashboard labels, CLI logs, and other user-facing sentinel renderings must
  show semantic meaning instead of raw numeric shapes; for example,
  `busy_threshold=-1` is unconditional keepalive mode, not `-1%` utilization.
- Dashboard source and packaged static assets must stay self-contained:
  no Google Fonts, CDN, remote CSS/JS/font/image imports, or other runtime
  network assets. Rebuild `src/keep_gpu/mcp/static/` after dashboard changes.
- Dashboard telemetry refresh must stay low-power: manual refresh by default,
  auto-refresh only after explicit user opt-in, and polling paused while the
  browser tab is hidden. Keep labels, cadence, tests, and packaged static assets
  in sync when changing refresh behavior.
- Single-GPU keep workload iteration counts must be positive integers (`relu_iterations` for CUDA, `iterations` for ROCm/Mac M); reject invalid values before keep loops so no public path can create a silent no-op keeper or late background thread crash.
- CUDA workload tuning must use the `relu_iterations` public keyword. Do not
  reintroduce the legacy `matmul_iterations` alias without an intentional,
  documented API decision.
- ROCm optional allocation retry counts must be `None` or positive plain
  integers; reject invalid values before worker startup so background retry
  loops cannot crash with type errors after `keep()` returns.
- Direct CUDA/ROCm single-GPU `rank` values must be plain integer visible
  device ordinals validated against the current visible device count during
  construction, before `torch.device`, backend `set_device`, telemetry, or keep
  worker startup can target an invalid ordinal.
- Treat public `gpu_ids` as visible device ordinals after user-supplied CUDA or ROCm visibility filtering. Do not rewrite visibility masks inside KeepGPU command paths; reject explicit CUDA/ROCm ordinals outside the current visible device count before starting keep workers.
- JSON-RPC and MCP tool `start_keep` requests with explicit `gpu_ids` outside
  the current visible device count are public invalid-parameter errors
  (`-32602` for direct JSON-RPC, MCP tool `isError=true`), not internal server
  failures.
- Keep CUDA telemetry aligned with visible CUDA ordinals:
  `get_gpu_utilization(index)` receives the visible rank used by
  `CudaGPUController`, and `gpu_monitor.py` resolves `CUDA_VISIBLE_DEVICES`
  ASCII numeric/UUID tokens to the correct NVML handle. UUID token resolution
  must preserve NVML string/bytes lookup fallbacks. If that mapping is
  malformed, duplicate/equivalent, ambiguous, or unsupported, return `None`
  without partial NVML handle queries so eco-safe backoff applies instead of
  falling back to a possibly wrong physical index.
- Keep ROCm telemetry aligned with visible ROCm ordinals: resolve `ROCR_VISIBLE_DEVICES` as the base mask and one matching `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` ASCII numeric overlay before querying ROCm SMI. If the mapping is malformed, conflicting, unsupported, or out of range, return unavailable utilization rather than querying a guessed SMI index.
- Keep GPU listing IDs aligned with start APIs: `list_gpus`/`/api/gpus`
  must expose `id` as the visible ordinal users can pass as `gpu_ids`;
  physical/vendor identifiers belong in explicit metadata fields such as
  `physical_id` and must not be accepted implicitly as selection IDs. CUDA NVML
  records must only be returned when Torch CUDA can address the same visible
  ordinal set, and malformed or duplicate CUDA visibility masks must not fall
  back to guessed Torch records; do not advertise NVML-only devices that
  controller startup cannot use. ROCm records must likewise be limited to
  visible ordinals that `torch.cuda.set_device()` can select; nullable ROCm
  memory fields mean memory telemetry is unavailable after successful
  selection, not that the device is unstartable.
- `KeepGPUServer.list_gpus()` must validate GPU records before any REST,
  direct JSON-RPC, or MCP response advertises them: required `id`/`visible_id`
  fields are matching plain non-negative integers, visible ordinals are unique,
  `platform`/`name` are strings, `memory_total` and `memory_used` are integers
  or null, and `utilization` is finite numeric or null. Malformed records from
  telemetry helpers are internal server failures.
- Keep GPU listing platform precedence aligned with controller platform
  detection: HIP/ROCm torch builds must prefer ROCm listing over NVML CUDA
  listing, and must not fall back to NVML CUDA records when torch's active
  runtime is HIP.
- Global controller startup must fail clearly when GPU selection resolves to zero or duplicate devices; do not create silent no-op or duplicate-worker keep sessions.
- Avoid scattering platform-specific branching across unrelated modules; prefer one clear decision path then platform-specific controller classes.
- Preserve simple controller flow: global controller orchestrates per-GPU controllers; single-GPU controllers handle device-level keep/release loops.

### Platform and Dependency Rules

- CUDA telemetry should rely on `nvidia-ml-py` (imported as `pynvml` module), not the deprecated standalone `pynvml` package.
- ROCm support is optional and should remain guarded. Do not depend on a
  non-PyPI `rocm-smi` distribution in package extras; treat `rocm_smi` as
  system-provided by the ROCm stack, and imports must fail gracefully on
  non-ROCm machines.
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
  - `pytest --run-large-memory -m large_memory` (only on machines where large VRAM allocations are acceptable)
- Respect existing pytest markers:
  - `rocm` for ROCm-only tests
  - `large_memory` for opt-in heavy VRAM tests; these are skipped unless `--run-large-memory` is supplied
- Keep broad validation matrices in the utility or controller test that owns
  the contract; CLI, REST, JSON-RPC, and MCP wrapper tests should use
  representative smoke cases plus side-effect guards instead of repeating every
  edge case.
- Run `pre-commit run --all-files` before final push.

### Documentation and Build Expectations

- Update docs when changing CLI flags, controller behavior, platform support, or MCP methods.
- Keep README and `docs/` guidance aligned for install and usage flows.
- Treat `pyproject.toml`, `.github/workflows/`, `docs/requirements.txt`, and
  `web/dashboard/package.json` as the canonical build/test command surfaces;
  do not reintroduce legacy root `Makefile` or `requirements_dev.txt`
  scaffolding for removed `setup.py`/Sphinx flows. Keep `MANIFEST.in` limited
  to source-distribution inclusions that `pyproject.toml` cannot express.
- Keep Ruff settings canonical in `pyproject.toml`; do not add a standalone
  `ruff.toml` unless the full configuration is intentionally migrated there.
- Validate docs with:
  - `mkdocs build` for static build verification
  - `mkdocs serve` for local preview when editing rendered content
- Do not commit generated `site/` output unless explicitly requested.

### Scope and Release Hygiene

- Keep commits narrow and module-focused (especially when touching controllers and platform utilities).
- Do not bump versions, move tags, or alter release metadata unless explicitly requested.
- When fixing review comments, prefer minimal diffs that address the exact finding and keep existing behavior stable.
