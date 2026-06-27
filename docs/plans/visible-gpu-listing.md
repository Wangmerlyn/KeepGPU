# Visible GPU Listing Plan

## Background

Public start APIs accept visible GPU ordinals: `0..torch.cuda.device_count()-1`
after environment masks such as `CUDA_VISIBLE_DEVICES`. The GPU listing path
currently forwards telemetry IDs from the backend probe. On the NVML path those
IDs are physical NVML indexes, so a masked process can display IDs that users
cannot pass back to `start_keep`.

## Goal

Make listed GPU IDs match the IDs accepted by the CLI, REST, JSON-RPC, and MCP
start paths. Preserve physical/vendor identity only as explicit metadata so the
dashboard and agents can explain the mapping without inviting invalid starts.

## Design

- Keep `gpu.id` as the public start-compatible visible ordinal.
- Add `visible_id` equal to `id` for explicitness in telemetry consumers.
- Add `physical_id` only when a backend can safely identify the underlying
  vendor/physical index.
- Honor `CUDA_VISIBLE_DEVICES` in NVML telemetry. Numeric masks map visible
  ordinals to physical NVML indexes; an empty mask returns no CUDA GPUs.
- Leave torch fallback and MPS semantics as visible ordinals.
- Keep selection APIs unchanged: `gpu_ids` remain visible ordinals only.
- Update dashboard labels and docs so humans type the visible ID, not the
  physical metadata.

## Todo

- [x] Add failing backend tests for masked NVML listing IDs.
- [x] Add failing dashboard helper tests for visible-first GPU labels.
- [x] Implement visible-ordinal GPU info output.
- [x] Update dashboard UI source and bundled static asset.
- [x] Update README, docs, and `AGENTS.md`.
- [x] Run targeted tests, full tests, docs build, pre-commit, and local subagent
      review.
- [ ] Open PR, resolve remote review/checks, squash merge, and clean up the
      worktree branch.
