# ROCm Mask Unknown Monitor Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent ROCm utilization telemetry from querying guessed physical SMI indexes when an explicit ROCm visibility mask is present but ROCm SMI cannot report monitor-device count.

**Architecture:** Keep the decision centralized in `rocm_visibility.py`. No-mask default mapping may still use the visible rank when the monitor count is unknown, but explicit `ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, or ROCm `CUDA_VISIBLE_DEVICES` masks must resolve to unavailable telemetry unless their physical IDs can be checked against a known monitor count.

**Tech Stack:** Python, pytest, ROCm SMI visibility helpers, KeepGPU ROCm controller and GPU listing tests, MkDocs.

---

## Task 1: Reject Unverifiable Explicit ROCm Masks

**Files:**
- Modify: `src/keep_gpu/utilities/rocm_visibility.py`
- Modify: `tests/rocm_controller/test_rocm_utilization.py`
- Modify: `tests/utilities/test_gpu_info.py`
- Modify: `AGENTS.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/guides/cli.md`
- Create: `docs/plans/rocm-mask-unknown-monitor-count.md`

- [x] **Step 1: Write failing ROCm controller regression**

Add a test showing that an explicit HIP mask is unavailable when `rsmi_num_monitor_devices()` fails:

```python
def test_rocm_utilization_treats_mask_as_unavailable_when_monitor_count_unknown(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy = DummyRocmSMI(count_error=RuntimeError("cannot count ROCm devices"))
    controller = _controller_with_dummy_smi(monkeypatch, rank=0, dummy=dummy)

    assert controller._query_utilization() is None
    assert dummy.queried_indexes == []
```

- [x] **Step 2: Write failing GPU listing regression**

Add GPU listing coverage so `/api/gpus` and `list_gpus` inherit the same unavailable-utilization behavior:

```python
def test_get_gpu_info_rocm_masked_utilization_unknown_when_monitor_count_unknown(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "9")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    dummy_rocm, _ = _install_rocm_gpu_info_mocks(
        monkeypatch,
        count=1,
        smi_count_error=RuntimeError("cannot count ROCm devices"),
        util_by_index={9: 99},
    )

    infos = gpu_info.get_gpu_info()

    assert len(infos) == 1
    assert infos[0]["platform"] == "rocm"
    assert infos[0]["utilization"] is None
    assert "physical_id" not in infos[0]
    assert dummy_rocm.queried_indexes == []
```

- [x] **Step 3: Verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/rocm_controller/test_rocm_utilization.py -q -k 'monitor_count_unknown'
PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q -k 'monitor_count_unknown'
```

Expected: both tests fail because explicit masks currently resolve to the numeric mask value when monitor count is `None`.

- [x] **Step 4: Implement minimal helper fix**

Change ROCm visibility resolution so explicit physical masks cannot be trusted when monitor count is unknown:

```python
def _physical_ids_available(
    physical_ids: Tuple[int, ...],
    monitor_count: Optional[int],
) -> bool:
    if monitor_count is None:
        return False
    return all(physical_id < monitor_count for physical_id in physical_ids)
```

Leave the no-mask default path unchanged so `visible_rank` can still map to itself when no explicit mask needs validation.

- [x] **Step 5: Update docs and AGENTS**

Document that explicit ROCm visibility masks are unavailable when ROCm SMI cannot report a monitor count, because KeepGPU cannot validate physical SMI indexes without guessing.

- [x] **Step 6: Verify GREEN and PR**

Run:

```bash
PYTHONPATH=src pytest tests/rocm_controller/test_rocm_utilization.py -q
PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q
PYTHONPATH=src pytest tests/mcp tests/utilities/test_gpu_info.py -q
PYTHONPATH=src pytest tests -q
PYTHONPATH=src mkdocs build --strict
pre-commit run --all-files --show-diff-on-failure
git diff --check
```

Then push the branch, open a PR, run local subagent review, address every comment, wait for hosted checks, and squash-merge only after all review threads are resolved.

Verification:
- `PYTHONPATH=src pytest tests/rocm_controller/test_rocm_utilization.py -q -k 'monitor_count_unknown'`
- `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q -k 'monitor_count_unknown'`
- `PYTHONPATH=src pytest tests/rocm_controller/test_rocm_utilization.py -q`
- `PYTHONPATH=src pytest tests/utilities/test_gpu_info.py -q`
- `PYTHONPATH=src pytest tests/mcp tests/utilities/test_gpu_info.py -q`
- `PYTHONPATH=src HIP_VISIBLE_DEVICES=9 python - <<'PY' ...`
- `PYTHONPATH=src pytest tests -q`
- `PYTHONPATH=src mkdocs build --strict`
- `pre-commit run --all-files --show-diff-on-failure`
- `git diff --check`
