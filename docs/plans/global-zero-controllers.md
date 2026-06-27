# Global Zero-Controller Guard Plan

## Background

`GlobalGPUController(gpu_ids=None)` expands to all visible CUDA devices. On a
no-GPU CUDA-like environment, `torch.cuda.device_count()` can return `0`, leaving
`self.controllers` empty. `keep()` then iterates over no workers and reports
success even though no GPU is being kept warm. Explicit `gpu_ids=[]` has the
same zero-controller shape.

## Goal

Fail clearly before startup when a global session would manage zero GPUs, so the
Python API, service API, and docs never imply a keepalive session is active when
no device-level worker exists.

## Design

- Treat an explicit empty `gpu_ids` list as invalid public input in the shared
  session-config validator.
- Keep `gpu_ids=None` as the "all visible GPUs" sentinel.
- After platform-specific expansion in `GlobalGPUController`, reject an empty
  resolved GPU list before constructing per-device controllers.
- Surface `ValueError` through existing JSON-RPC/REST/CLI error paths instead
  of inventing a new exception type.
- Update docs and `AGENTS.md` to state that an empty resolved selection is a
  startup error.

## Todo

- [x] Add failing no-GPU tests for explicit `gpu_ids=[]` and CUDA
      auto-discovery returning zero devices.
- [x] Add service-level coverage proving empty `gpu_ids` is rejected before a
      session is registered.
- [x] Implement the shared empty-list validator and resolved-selection guard.
- [x] Update `AGENTS.md`, README, Python/API/MCP docs with the empty-selection
      startup behavior.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local
      subagent review before opening the PR.
