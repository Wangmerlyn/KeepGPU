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

### Architecture Boundaries

- Keep platform detection and environment probing centralized in `src/keep_gpu/utilities/platform_manager.py`.
- Keep GPU telemetry helpers in `src/keep_gpu/utilities/gpu_info.py` and related utility modules.
- Keep public session input validation centralized in `src/keep_gpu/utilities/session_config.py`; CLI, Python, REST, and JSON-RPC entry points must share the same contract.
- Keep human VRAM parsing centralized in `src/keep_gpu/utilities/humanized_input.py`; public integer values and digit-only strings mean bytes, while controllers may convert to internal tensor element counts.
- Hardware probes must clean up vendor libraries after detection (for example, NVML shutdown and ROCm SMI shutdown after init).
- Keep lifecycle state truthful: a session is removed only after release succeeds; timed-out or failed stops must stay visible with state and error details.
- Keep service daemon ownership safe: no stop, force-stop, or fallback path may signal a PID unless the auto-start ownership record verifies the running process.
- Treat custom `job_id` values as reserved from the moment startup begins; duplicate starts must fail before another controller can begin keep-alive work.
- Keep custom `job_id` validation centralized in `session_config.py`: only `None` means omitted/all-sessions, and custom IDs must be non-empty URL-path-safe strings before any session state changes.
- Stop requests must not miss starting sessions; wait for startup to settle before returning `not found` or taking a stop-all snapshot.
- Stop-all may release independent sessions concurrently, but must not duplicate release work for `stopping` sessions and must keep deterministic additive result fields.
- Keep utilization backoff eco-safe: valid `busy_threshold` values are `-1` or `0..100`; when telemetry is unavailable and `busy_threshold >= 0`, controllers should sleep instead of running keepalive compute. Only `busy_threshold=-1` is the explicit unconditional mode.
- Keep CUDA telemetry aligned with visible CUDA ordinals: `get_gpu_utilization(index)` receives the visible rank used by `CudaGPUController`, and `gpu_monitor.py` resolves `CUDA_VISIBLE_DEVICES` numeric/UUID tokens to the correct NVML handle. If that mapping is ambiguous or unsupported, return `None` rather than falling back to a possibly wrong physical index.
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
