# Docs Platform Parity Plan

## Background

KeepGPU supports CLI, Python controllers, and MCP as first-class surfaces across
CUDA, ROCm, and Mac M/MPS paths. A few public docs entry points still used
CUDA-only framing or described MCP as experimental.

## Goal

Keep the public docs concise while making the platform and interface story match
the current product surface.

## Tasks

- [x] Create an isolated worktree branch from latest `main`.
- [x] Add a RED docs guard for stale CUDA-only and MCP-experimental wording.
- [x] Update overview, getting-started, architecture, contributing, and AGENTS
      wording.
- [x] Run docs/tests.
- [ ] Run local subagent review.
- [ ] Open PR, resolve review comments, wait for clean checks, squash merge, and
      clean the worktree.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_public_docs_do_not_regress_to_cuda_only_or_experimental_mcp -q`,
  failed because `docs/index.md` still used `low-cost CUDA workloads`.
- GREEN:
  the same command passed after the public docs used platform-neutral wording
  and described MCP as a first-class interface.
