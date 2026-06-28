# Controller Startup Failures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure KeepGPU never reports a CUDA or ROCm keep session as started when the worker thread immediately fails during backend startup.

**Architecture:** Add a small startup handshake between `keep()` and the background worker for CUDA and ROCm controllers. The worker should signal either "backend startup is ready" after `torch.cuda.set_device(...)` succeeds or return the startup exception to `keep()` before public APIs report success. This keeps the service layer honest because `GlobalGPUController.keep()` will fail and roll back instead of registering an active session.

**Tech Stack:** Python, pytest, PyTorch controller classes, KeepGPU MCP/JSON-RPC service.

---

## Background

CUDA and ROCm single-GPU controllers currently call `torch.cuda.set_device(...)`
inside the daemon worker thread. If that backend startup fails, `keep()` has
already returned and the service can register an active job even though no
keepalive worker is running.

## Solution

- Add RED tests that reproduce CUDA and ROCm startup failure without requiring a
  GPU by monkeypatching `torch.cuda.set_device`.
- Add a JSON-RPC/service regression test showing failed CUDA worker startup must
  not leave an active session.
- Implement a minimal startup handshake in CUDA and ROCm controllers.
- Document the invariant in `AGENTS.md` and API guidance.

## Todo

- [x] Run targeted controller/service baseline.
- [x] Add failing controller startup tests.
- [x] Add failing service no-active-session regression test.
- [x] Verify the new tests fail on current behavior.
- [x] Implement the startup handshake.
- [x] Verify focused tests pass.
- [x] Update docs and `AGENTS.md`.
- [x] Run broader tests, docs build, pre-commit, and local subagent review.
- [ ] Open a PR, resolve review comments, squash merge, and clean up the branch.

## Verification

- `PYTHONPATH=$PWD/src pytest tests/cuda_controller tests/rocm_controller tests/mcp/test_server.py -q`
- `PYTHONPATH=$PWD/src pytest tests -q`
- `PYTHONPATH=$PWD/src mkdocs build`
- `pre-commit run --all-files`
