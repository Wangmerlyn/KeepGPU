# CLI RPC Error Message Validation Plan

> **For agentic workers:** Follow this plan task-by-task. Use test-first changes for code behavior.

**Goal:** Reject malformed JSON-RPC error objects before they can be mistaken for real service failures.

**Architecture:** Keep `_rpc_call()` responsible only for generic JSON-RPC envelope validation. A service error is a `ServiceRPCError` only when the error object has a usable integer code, if present, and a string `message`; malformed error envelopes remain `ServiceResponseError` and must not trigger daemon rollback.

**Tech Stack:** Python, Typer CLI, pytest, MkDocs.

---

### Task 1: Reproduce the malformed error envelope

**Files:**
- Modify: `tests/test_cli_service_commands.py`

- [ ] Add a test showing `_rpc_call()` raises `ServiceResponseError` when `error.message` is missing or not a string.
- [ ] Add a test showing `keep-gpu start` does not stop a just-auto-started daemon when the `start_keep` response has `code=-32000` but a malformed `error.message`.
- [ ] Run the focused tests and confirm they fail for the current behavior.

### Task 2: Validate `error.message`

**Files:**
- Modify: `src/keep_gpu/cli.py`

- [ ] Require JSON-RPC error objects to contain a string `message` before raising `ServiceRPCError`.
- [ ] Leave method-specific result validation at CLI call sites.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Keep docs aligned

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/reference/cli.md`

- [ ] Document that malformed JSON-RPC error objects include missing or non-string `error.message`.
- [ ] Document that malformed service envelopes do not trigger startup-unavailable daemon rollback.

### Task 4: Verify and publish

- [ ] Run `PYTHONPATH=src pytest tests/test_cli_service_commands.py -q`.
- [ ] Run `PYTHONPATH=src pytest tests -q`.
- [ ] Run `mkdocs build --strict`.
- [ ] Run `pre-commit run --all-files --show-diff-on-failure`.
- [ ] Run `git diff --check`.
- [ ] Commit, push, open a PR, run local subagent review, resolve comments, wait for hosted checks, and squash merge.
