# CUDA Mask Fallback Plan

## Goal

Do not advertise Torch fallback CUDA records when `CUDA_VISIBLE_DEVICES` is malformed.

## Scope

- Preserve valid-mask Torch fallback behavior for start-compatible visible ordinals.
- Treat malformed/duplicate CUDA masks as unavailable listing state, not as "all hidden" or "guess with Torch".
- Keep the change inside `gpu_info.py` with focused utility tests.

## Tasks

- [x] Add failing tests for malformed CUDA masks with Torch fallback available.
- [x] Add a small mask-state helper that distinguishes hidden-all, valid tokens, and invalid masks.
- [x] Use the invalid-mask state to skip CUDA Torch fallback while preserving MPS fallback.
- [x] Update `AGENTS.md` and docs after tightening the listing contract.
- [x] Run targeted/full tests, docs build, and pre-commit.
- [x] Address local review finding for equivalent duplicate mask tokens such as `0,00`.
- [x] Address hosted review finding for Unicode digit masks such as `\u00b2`.
- [x] Run local review; PR checks and squash merge are tracked on GitHub.
