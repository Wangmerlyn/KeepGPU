# REST Session Empty GPUs Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make REST session docs distinguish invalid explicit empty `gpu_ids` from startup-unavailable empty GPU listings.

**Architecture:** This is a docs-contract fix only. The existing REST behavior and tests already reject explicit empty selections as validation errors and reserve `503` for startup-unavailable listing/probe results.

**Tech Stack:** Markdown docs, pytest documentation guards, MkDocs.

---

### Task 1: Document the Existing REST Contract

**Files:**
- Modify: `tests/test_package_metadata.py`
- Modify: `docs/reference/cli.md`
- Modify: `AGENTS.md`
- Create: `docs/plans/rest-session-empty-gpus-doc.md`

- [x] **Step 1: Add a failing docs guard**

Add `test_rest_session_docs_distinguish_empty_gpu_ids_from_empty_listing` to assert that `docs/reference/cli.md` says an explicit empty `gpu_ids` list is invalid, says an empty validated GPU listing is `503`, and no longer says "a valid empty list is `503`".

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_metadata.py::test_rest_session_docs_distinguish_empty_gpu_ids_from_empty_listing -q
```

Expected: fails because the current CLI reference still uses the stale wording.

- [x] **Step 3: Update docs and agent guidance**

Rewrite the `/api/sessions` POST reference row and MCP/service guide to distinguish explicit empty selections from empty validated listings. Update `AGENTS.md` with the same preservation rule.

- [x] **Step 4: Verify GREEN and docs build**

Run:

```bash
PYTHONPATH=src pytest tests/test_package_metadata.py::test_rest_session_docs_distinguish_empty_gpu_ids_from_empty_listing -q
PYTHONPATH=src pytest tests/test_package_metadata.py -q
PYTHONPATH=src pytest tests/mcp/test_http_api.py -q -k 'empty_gpu_ids or startup_unavailable or validates_gpu_ids'
PYTHONPATH=src mkdocs build --strict
```

- [ ] **Step 5: Review and PR**

Run pre-commit and `git diff --check`, then open a focused PR for local subagent review.
