# List GPUs Enumeration Unavailable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface CUDA/ROCm visible-device enumeration failures as startup-unavailable list-gpus errors instead of successful empty GPU lists.

**Architecture:** Keep the root cause in `gpu_info`: after CUDA/ROCm is reported available, failed `torch.cuda.device_count()` or `torch.cuda.current_device()` means enumeration is unavailable, while a valid zero count remains an empty listing. `torch.cuda.is_available()` probe failures stay soft so non-CUDA fallbacks such as MPS can still work. Existing service, JSON-RPC, MCP, REST, and CLI layers already know how to classify `DeviceEnumerationUnavailableError`.

**Tech Stack:** Python, pytest, KeepGPU utility/service layers, MkDocs.

---

## Task 1: Propagate Enumeration Failures from GPU Listing

**Files:**
- Modify: `src/keep_gpu/utilities/gpu_info.py`
- Modify: `tests/utilities/test_gpu_info.py`
- Modify: `tests/mcp/test_server.py`
- Modify: `AGENTS.md`
- Modify: `docs/reference/cli.md`
- Create: `docs/plans/list-gpus-enumeration-unavailable.md`

- [x] **Step 1: Write failing regressions**

Flip the CUDA NVML count-failure test to expect `DeviceEnumerationUnavailableError`, add ROCm count-failure coverage, and add a service test that uses the real `gpu_info.get_gpu_info()` helper path through `KeepGPUServer.list_gpus()`.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q -k 'count_fails or enumeration_unavailable'
PYTHONPATH=src pytest tests/mcp/test_server.py -q -k 'list_gpus and enumeration'
```

Expected: fails because `get_gpu_info()` currently swallows count failures and returns an empty list.

- [x] **Step 3: Implement root-cause fix**

Raise `DeviceEnumerationUnavailableError` when post-availability CUDA/ROCm `torch.cuda.device_count()` or `torch.cuda.current_device()` fails in NVML, ROCm, or torch fallback listing paths. Re-raise that error through `get_gpu_info()` instead of falling back to other listing modes.

- [x] **Step 4: Update docs and agent guidance**

Document that list-gpus surfaces CUDA/ROCm enumeration failures as startup-unavailable instead of empty lists.

- [x] **Step 5: Verify GREEN and PR**

Run targeted utility/service tests, broader MCP/GPU-info tests, full tests, docs build, pre-commit, and local subagent review before opening the PR.

Verification:
- `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q -k 'cuda_nvml_raises_when_torch_current_device_fails'`
- `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q -k 'availability_fails or current_device_fails or count_fails or enumeration_unavailable or mps_fallback'`
- `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q`
- `PYTHONPATH=src pytest tests/mcp/test_server.py -q -k 'list_gpus or DeviceEnumerationUnavailableError or startup_unavailable'`
- `PYTHONPATH=src pytest tests/mcp/test_http_api.py -q -k 'api_gpus or list_gpus or startup_unavailable or enumeration_unavailable'`
- `PYTHONPATH=src pytest tests -q`
