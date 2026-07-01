# API GPUs Noncanonical Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure noncanonical `/api/gpus` HTTP routes with params or query components return structured `404 Unknown endpoint` responses without invoking GPU telemetry.

**Architecture:** Keep the fix centralized in `_JSONRPCHandler._is_noncanonical_api_route()` so `/api/gpus` params/query requests are rejected before telemetry dispatch. Preserve exact canonical `/api/gpus` dispatch and existing session path component validation behavior.

**Tech Stack:** Python `http.server` handler, `urllib.parse.urlparse`, pytest HTTP integration tests.

---

### Task 1: Regression Test

**Files:**
- Modify: `tests/mcp/test_http_api.py`

- [x] **Step 1: Add a minimal parametrized test**

Add a test near the existing API route 404 tests:

```python
@pytest.mark.parametrize("path", ["/api/gpus?bad=query", "/api/gpus;bad"])
def test_http_get_api_gpus_noncanonical_route_returns_json_404_without_listing(path):
    server = make_server()
    list_calls = []

    def fail_list_gpus():
        list_calls.append(True)
        raise AssertionError("list_gpus should not run for noncanonical route")

    server.list_gpus = fail_list_gpus  # type: ignore[method-assign]
    httpd, thread, base = _start_http_server(server)

    try:
        status_code, payload = _request_json("GET", f"{base}{path}")
    finally:
        httpd.shutdown()
        httpd.server_close()
        server.shutdown()
        thread.join(timeout=2)

    assert status_code == 404
    assert payload == {"error": {"message": "Unknown endpoint"}}
    assert list_calls == []
```

- [x] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=src pytest tests/mcp/test_http_api.py::test_http_get_api_gpus_noncanonical_route_returns_json_404_without_listing -q
```

Expected: fail because current routing dispatches canonical-looking `/api/gpus` params/query requests to `list_gpus()`.

### Task 2: Minimal Fix

**Files:**
- Modify: `src/keep_gpu/mcp/server.py`

- [x] **Step 1: Fix API noncanonical detection**

Update `_is_noncanonical_api_route()` so `/api/gpus` with `parsed.params`, `parsed.query`, or `parsed.fragment` is noncanonical, while exact canonical `/api/gpus` without those components continues normally.

- [x] **Step 2: Run GREEN**

Run:

```bash
PYTHONPATH=src pytest tests/mcp/test_http_api.py::test_http_get_api_gpus_noncanonical_route_returns_json_404_without_listing -q
```

Expected: pass.

### Task 3: Verification and Commit

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/plans/api-gpus-noncanonical.md`
- Modify: `tests/mcp/test_http_api.py`
- Modify: `src/keep_gpu/mcp/server.py`

- [x] **Step 1: Run focused HTTP API tests**

Run:

```bash
PYTHONPATH=src pytest tests/mcp/test_http_api.py -q
```

Expected: pass.

- [x] **Step 2: Run whitespace diff check**

Run:

```bash
git diff --check
```

Expected: pass with no output.

- [x] **Step 3: Update agent contract**

Document that exact API endpoints such as `/api/gpus` do not accept params,
query strings, or fragments unless explicitly documented by the handler.

- [x] **Step 4: Commit**

Run:

```bash
git add docs/plans/api-gpus-noncanonical.md tests/mcp/test_http_api.py src/keep_gpu/mcp/server.py
git commit -m "fix(mcp): reject noncanonical api gpu routes"
```
