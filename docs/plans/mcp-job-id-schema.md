# MCP Job ID Schema Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep MCP tool schemas aligned with the shared public `job_id` validation contract.

**Architecture:** `session_config.py` remains the single source for public session input validation. The MCP server imports a public job-id pattern string from that utility module and applies it to every `job_id` tool schema, so clients see the same non-empty URL-path-safe contract that runtime calls enforce.

**Tech Stack:** Python, MCP JSON-RPC handlers, pytest, MkDocs.

---

## Task 1: Advertise the Shared Job ID Contract in MCP Schemas

**Files:**
- Modify: `src/keep_gpu/utilities/session_config.py`
- Modify: `src/keep_gpu/mcp/server.py`
- Modify: `tests/mcp/test_server.py`
- Modify: `docs/guides/mcp.md`
- Modify: `AGENTS.md`
- Create: `docs/plans/mcp-job-id-schema.md`

- [x] **Step 1: Write a failing schema regression**

Extend `test_mcp_tools_list_exposes_keepgpu_actions` so `start_keep`, `stop_keep`, and `status` all require:

```python
{
    "type": ["string", "null"],
    "minLength": 1,
    "pattern": JOB_ID_PATTERN_TEXT,
}
```

Run:

```bash
PYTHONPATH=src pytest tests/mcp/test_server.py -q -k tools_list_exposes_keepgpu_actions
```

Expected before the fix: the test fails because MCP schemas expose only `type`.

- [x] **Step 2: Expose the runtime pattern text**

Add `JOB_ID_PATTERN_TEXT = r"^[A-Za-z0-9._~-]+$"` in `session_config.py` and compile `_JOB_ID_PATTERN` from it.

- [x] **Step 3: Reuse the pattern in MCP schemas**

Import `JOB_ID_PATTERN_TEXT` in `mcp/server.py` and add `minLength` plus `pattern` to every MCP `job_id` schema.

- [x] **Step 4: Update docs and agent contract**

Mention that MCP schemas advertise the same URL-path-safe custom ID contract that runtime validation enforces.

- [x] **Step 5: Verify**

Run:

```bash
PYTHONPATH=src pytest tests/mcp/test_server.py -q -k tools_list_exposes_keepgpu_actions
PYTHONPATH=src pytest tests/mcp/test_server.py tests/utilities/test_session_config.py -q
PYTHONPATH=src pytest tests -q
PYTHONPATH=src mkdocs build --strict
pre-commit run --all-files --show-diff-on-failure
git diff --check
```
