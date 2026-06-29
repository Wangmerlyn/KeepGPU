# Startup Enumeration Unavailable Plan

## Goal

Classify CUDA/ROCm visible-device enumeration failures as expected
startup-unavailable errors instead of internal failures.

## Background

`GlobalGPUController` selects CUDA/ROCm from platform detection and then asks
PyTorch for visible device ordinals. When that runtime enumeration fails or
changes during startup, the service should report hardware/runtime
unavailability, not an arbitrary server bug.

The audited gaps were:

- Initial or child-controller `torch.cuda.device_count()` failures leaked as
  JSON-RPC `-32603`.
- Late child-controller visible-count drift to zero/smaller counts leaked as
  plain `ValueError`.
- REST explicit `gpu_ids` starts returned `400` when no visible GPUs could be
  listed.
- REST explicit `gpu_ids` starts returned `500` if GPU listing raised a device
  enumeration error.
- Blocking CLI mode queried device count before `GlobalGPUController` could own
  omitted-GPU enumeration.

## Design

- Add `visible_torch_device_count()` and
  `DeviceEnumerationUnavailableError` in `platform_manager.py`.
- Use that helper in global, CUDA, and ROCm controller enumeration paths.
- Keep invalid explicit `gpu_ids` as invalid input when a visible set is known.
- Add `VisibleRankValidationError` so global startup classification does not
  depend on error-message text.
- Map enumeration failures and late visible-rank drift to
  `NoGPUAvailableError`, which flows to JSON-RPC `-32000`, REST `503`, and MCP
  tool `isError=true`.
- Map REST GPU-listing enumeration failures to the same structured `503`.
- Let blocking CLI pass omitted `gpu_ids=None` through to `GlobalGPUController`
  instead of probing the device count for logging.

## Docs

- `AGENTS.md`: startup-unavailable and CLI omitted-GPU contracts.
- `docs/reference/cli.md`: blocking CLI omitted-GPU wording and JSON-RPC
  startup-unavailable wording.
- `docs/guides/mcp.md`: direct JSON-RPC and MCP tool startup-unavailable
  behavior.

## Verification

- Targeted startup/CLI/session-config shard: `8 passed`; REST listing-failure
  shard: `3 passed`.
- Affected suites (`test_cli_thresholds.py`, MCP server/API,
  global/CUDA/ROCm controllers, utilities): `436 passed, 7 skipped`.
- Full suite: `664 passed, 11 skipped`.
- `pre-commit run --all-files`: passed.
- `PYTHONPATH=$PWD/src mkdocs build`: passed with existing Material warning and
  existing unnav'd plan notices.
- `git diff --check`: passed.
