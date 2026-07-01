# CUDA Visible Mask Parity Plan

## Background

CUDA accepts abbreviated GPU UUID prefixes when they uniquely identify a device,
and stops parsing `CUDA_VISIBLE_DEVICES` at the first invalid token such as `-1`.
KeepGPU currently treats UUID tokens as exact-only and treats any `-1` token as
an invalid whole mask.

## Goal

Keep `list_gpus`, REST validation, and runtime telemetry aligned with CUDA and
PyTorch visible ordinals for valid prefix masks and valid-prefix-before-`-1`
masks, while preserving the existing safe behavior for ambiguous or malformed
masks.

## Solution

- Add RED tests for unique UUID-prefix masks in GPU listing and telemetry.
- Add RED tests for `CUDA_VISIBLE_DEVICES=0,2,-1,1` exposing only `[0, 2]`.
- Resolve UUID tokens by exact UUID lookup first, then by unique NVML UUID
  prefix enumeration.
- Stop CUDA mask parsing at `-1` after accepting any valid preceding tokens.

## Verification

- Focused CUDA visibility, GPU info, and GPU monitor tests.
- Relevant utility and MCP list-gpu tests.
- Full Python tests, pre-commit, docs build, and diff whitespace check before PR.
