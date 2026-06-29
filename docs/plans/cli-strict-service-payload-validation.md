# CLI Strict Service Payload Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten CLI service result validation so malformed method-specific list items and required records become clean JSON errors instead of successful machine-readable output.

**Architecture:** Keep `_rpc_call()` limited to generic JSON-RPC envelope validation. Extend the existing CLI-side `status`, `stop_keep`, and `list_gpus` validators to validate only the stable fields promised by the service contract while still allowing extra fields for forward compatibility.

**Tech Stack:** Python, Typer CLI, pytest, mkdocs, pre-commit.

---

## Background

The first CLI payload validation pass rejects missing top-level method fields such as `active_jobs`, `stopped`, `errors`, and `gpus`. The remaining gap is that malformed entries inside those containers can still be printed as success: `{"active_jobs": [123]}`, `{"active": true, "job_id": "job-1"}`, `{"stopped": [123], ...}`, or `{"gpus": [{"name": "missing id"}]}`. That violates the CLI JSON contract for downstream automation.

## Task 1: Strict Method-Specific CLI Validators

**Files:**
- Modify: `src/keep_gpu/cli.py`
- Modify: `tests/test_cli_service_commands.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/reference/cli.md`
- Modify: `docs/guides/cli.md`

- [x] **Step 1: Write RED tests for malformed nested payloads**

Add focused tests in `tests/test_cli_service_commands.py` that currently fail because malformed nested payloads exit `0`:

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"active_jobs": [123]},
        {"active_jobs": [{"params": {}, "state": "active", "last_error": None}]},
        {"active_jobs": [{"job_id": "job-1", "state": "active", "last_error": None}]},
        {"active_jobs": [{"job_id": "job-1", "params": {}, "last_error": None}]},
        {"active_jobs": [{"job_id": "job-1", "params": {}, "state": "active", "last_error": 1}]},
    ],
)
def test_status_rejects_malformed_active_job_entries(monkeypatch, payload):
    ...
```

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"active": True, "job_id": "job-1"},
        {"active": True, "job_id": "job-1", "params": [], "state": "active", "last_error": None},
        {"active": True, "job_id": "job-1", "params": {}, "state": 1, "last_error": None},
        {"active": True, "job_id": "job-1", "params": {}, "state": "active", "last_error": 1},
    ],
)
def test_status_job_rejects_active_payload_missing_session_fields(monkeypatch, payload):
    ...
```

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"stopped": [123], "timed_out": [], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": [False], "failed": [], "errors": {}},
        {"stopped": [], "timed_out": [], "failed": [{}], "errors": {}},
        {"stopped": [], "timed_out": [], "failed": [], "errors": {"job-1": 2}},
    ],
)
def test_stop_job_rejects_malformed_job_id_lists_and_errors(monkeypatch, payload):
    ...
```

```python
@pytest.mark.parametrize(
    "payload",
    [
        {"gpus": [{"name": "missing id"}]},
        {"gpus": [{"id": True, "visible_id": 0, "platform": "cuda", "name": "GPU", "memory_total": None, "memory_used": None, "utilization": None}]},
        {"gpus": [{"id": 0, "visible_id": "0", "platform": "cuda", "name": "GPU", "memory_total": None, "memory_used": None, "utilization": None}]},
        {"gpus": [{"id": 0, "visible_id": 0, "platform": 1, "name": "GPU", "memory_total": None, "memory_used": None, "utilization": None}]},
    ],
)
def test_list_gpus_rejects_records_missing_required_fields(monkeypatch, payload):
    ...
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'malformed_active_job_entries or active_payload_missing_session_fields or malformed_job_id_lists_and_errors or records_missing_required_fields'
```

Expected: tests fail because the current validators accept these nested malformed payloads.

- [x] **Step 3: Implement minimal strict validators**

In `src/keep_gpu/cli.py`:

- Add plain-type helpers such as `_is_plain_int()` and field validators for required string, object, optional string, optional integer, and optional numeric fields.
- Update `_validate_status_result()`:
  - all-session `active_jobs` must be a list of objects;
  - each active job must include `job_id: str`, `params: object`, `state: str`, and `last_error: str | None`;
  - single-job responses still require `active: bool` and `job_id: str`;
  - if `active` is `True`, require `params: object`, `state: str`, and `last_error: str | None`;
  - if `active` is `False`, do not require session detail fields.
- Update `_validate_stop_keep_result()` so `stopped`, `timed_out`, and `failed` are lists of strings, `errors` is an object whose values are strings, and optional `message` remains a string.
- Update `_validate_list_gpus_result()` so each GPU record includes `id` and `visible_id` as non-bool ints, `platform` and `name` as strings, and `memory_total`, `memory_used`, and `utilization` as nullable telemetry values.
- Reject non-finite numeric utilization values such as `NaN` and `Infinity`, because the CLI promises directly parseable JSON output.
- Keep extra result fields and extra per-record fields allowed.

- [x] **Step 4: Verify GREEN**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q -k 'malformed_active_job_entries or active_payload_missing_session_fields or inactive_payload_without_session_fields or malformed_job_id_lists_and_errors or records_missing_required_fields or outputs_single_decoded_json_object or service_stop_rejects_malformed'
PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q
```

- [x] **Step 5: Update docs and AGENTS**

Document that CLI service JSON commands reject malformed method-specific records, not just malformed top-level envelopes:

- `AGENTS.md`: clarify required nested record validation for `status`, `stop_keep`, and `list_gpus`.
- `README.md`, `docs/reference/cli.md`, and `docs/guides/cli.md`: note that malformed service envelopes and malformed method-specific result records are reported as single JSON error objects.

- [x] **Step 6: Run broad verification**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_cli_service_commands.py -q
PYTHONPATH=$PWD/src pytest tests -q
pre-commit run --all-files
PYTHONPATH=$PWD/src mkdocs build
git diff --check
```

- [x] **Step 7: Commit**

Commit with:

```bash
git add AGENTS.md README.md docs/reference/cli.md docs/guides/cli.md docs/plans/cli-strict-service-payload-validation.md src/keep_gpu/cli.py tests/test_cli_service_commands.py
git commit -m "fix(cli): validate service result records"
```
