# README Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `README.md` into a clean front door while keeping detailed CLI, Python, MCP, platform, and development guidance in published docs.

**Architecture:** Keep README as a compact product overview with links to existing docs. Move no behavior into new code paths and avoid creating a new docs page unless an existing destination is missing. Add a tiny metadata test so the README does not quietly grow back into a reference manual.

**Tech Stack:** Markdown, MkDocs, pytest, pre-commit.

---

### Task 1: Add README Shape Guard

**Files:**
- Modify: `tests/test_package_metadata.py`

- [x] **Step 1: Add a failing test**

Add this test after `test_readme_markdown_code_fences_are_balanced`:

```python
def test_readme_stays_a_compact_front_door():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    lines = [line for line in readme.splitlines() if line.strip()]

    assert len(lines) <= 130
    assert "### MCP and service API" not in readme
    assert "Platform installs at a glance" not in readme
    assert "docs/guides/mcp.md" in readme
    assert "docs/reference/cli.md" in readme
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_readme_stays_a_compact_front_door -q
```

Expected: FAIL because the current README has more than 130 nonblank lines.

### Task 2: Slim README and Keep Links

**Files:**
- Modify: `README.md`

- [x] **Step 1: Replace README with a compact front door**

Keep these sections only:

- `# Keep GPU`
- badges and one-sentence product pitch
- `## Why KeepGPU`
- `## Quick Start`
- `## Python`
- `## Service, Dashboard, and MCP`
- `## Documentation`
- `## Contributing`
- `## Citation`

The README must include:

- one blocking CLI example
- one service-mode example
- one Python snippet
- dashboard URL
- links to `docs/getting-started.md`, `docs/guides/cli.md`, `docs/guides/python.md`, `docs/guides/mcp.md`, `docs/reference/cli.md`, `docs/reference/api.md`, `docs/concepts/architecture.md`, and `docs/contributing.md`
- the existing BibTeX citation

The README must not include:

- full platform install matrix
- CLI flag reference tables or long validation rules
- JSON-RPC/REST protocol details
- dashboard lifecycle edge cases
- developer test matrices

- [x] **Step 2: Verify README size**

Run:

```bash
python - <<'PY'
from pathlib import Path
lines = [line for line in Path("README.md").read_text(encoding="utf-8").splitlines() if line.strip()]
print(len(lines))
assert len(lines) <= 130
PY
```

Expected: prints a number no greater than `130`.

### Task 3: Update Agent Guidance

**Files:**
- Modify: `AGENTS.md`

- [x] **Step 1: Add README guardrail**

Under Documentation Updates, add a bullet with this meaning:

```markdown
- Keep `README.md` as a concise front door. Put detailed CLI/API/MCP/platform
  contracts in `docs/` pages and link to them from README instead of repeating
  reference material.
```

### Task 4: Verify and Review

**Files:**
- Modify: `docs/plans/readme-slim.md`

- [x] **Step 1: Run targeted tests**

```bash
PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q
```

Expected: all tests in the file pass.

- [x] **Step 2: Run docs and formatting gates**

```bash
pre-commit run --all-files --show-diff-on-failure
mkdocs build --strict
```

Expected: both commands exit `0`. The existing upstream Material for MkDocs warning may appear.

- [x] **Step 3: Mark this plan complete**

Check off completed plan items in `docs/plans/readme-slim.md`.

- [x] **Step 4: Request local subagent code review**

Dispatch a local reviewer before opening the PR. Resolve any Critical or Important findings before pushing.
