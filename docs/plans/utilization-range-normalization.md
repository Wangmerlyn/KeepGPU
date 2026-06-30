# Utilization Range Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treat malformed or out-of-range GPU utilization as unavailable telemetry so non-negative backoff remains eco-safe.

**Architecture:** Keep one shared predicate in `session_config.py`. Normalize raw vendor readings at CUDA NVML, ROCm SMI, and GPU-listing ingress; reject malformed public `list_gpus` payloads at the CLI and MCP/REST service boundary.

**Tech Stack:** Python, pytest, Typer CLI, JSON-RPC/MCP service.

---

## Tasks

- [x] Add RED tests for CUDA/ROCm controller decisions with `-1`, `101`, and boolean utilization.
- [x] Add RED tests for NVML/ROCm telemetry helpers normalizing out-of-range vendor readings to `None`.
- [x] Add RED tests for CLI, JSON-RPC, and REST GPU-list payload validation rejecting out-of-range utilization fields.
- [x] Add shared utilization normalization in `src/keep_gpu/utilities/session_config.py`.
- [x] Route controller backoff, NVML monitor, ROCm monitor, and GPU listing through the shared helper.
- [x] Update `AGENTS.md` and user docs with the finite `0..100` or `null` utilization contract.
- [ ] Run targeted suites, full tests, docs build, pre-commit, local subagent review, PR checks, and squash merge.

## Verification

- Red: focused utilization shard failed with 16 expected behavior failures because `-1`/`101`/boolean utilization still flowed as real telemetry.
- Green: the same focused shard passed with `39 passed`.
