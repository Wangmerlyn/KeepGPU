# README Slimming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `README.md` as a clean front door while detailed CLI, Python, MCP, platform, and development guidance live in focused docs pages.

**Architecture:** README should provide a short pitch, one quick-start command, and links to existing docs. Existing docs already cover service mode, Python controllers, dashboard, MCP, references, contributing, and citation, so this change does not add a new page. A metadata test prevents README from drifting back into a reference manual.

**Tech Stack:** Markdown, MkDocs, pytest, pre-commit.

---

### Task 1: Tighten README Shape Guard

**Files:**
- Modify: `tests/test_package_metadata.py`

- [x] **Step 1: Add the stricter failing test**

`test_readme_stays_a_compact_front_door` must enforce:

```python
assert len(lines) <= 60
assert "## python" not in normalized_readme
assert "## service, dashboard, and mcp" not in normalized_readme
assert "### mcp and service api" not in normalized_readme
assert "platform installs at a glance" not in normalized_readme
assert "```bibtex" not in normalized_readme
assert "skillcheck" not in normalized_readme
assert not re.search(r"\]\(\.?\.?/?docs/", readme)
assert "https://keepgpu.readthedocs.io/en/latest/guides/mcp/" in readme
assert "https://keepgpu.readthedocs.io/en/latest/reference/cli/" in readme
assert "https://keepgpu.readthedocs.io/en/latest/citation/" in readme
```

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_readme_stays_a_compact_front_door -q
```

Expected: FAIL because the previous README still had standalone Python and
service/dashboard/MCP sections.

### Task 2: Slim README to a Front Door

**Files:**
- Modify: `README.md`

- [x] **Step 1: Keep only front-door content**

README keeps:

- title and badges
- one-sentence product pitch
- three `Why KeepGPU` bullets
- one install/start command block
- a compact interface table linking to existing docs
- short documentation, contributing, and citation links
- package-renderer-safe absolute links for docs, contributing, and citation
- only badges with live image URLs

README does not keep:

- standalone Python section
- standalone service/dashboard/MCP section
- service-mode command sequence
- Python controller snippet
- dashboard URL block
- full citation metadata
- platform install matrix
- CLI/API/MCP contract detail
- repository-relative `docs/...` links in README
- broken badges

- [x] **Step 2: Verify GREEN**

Run:

```bash
PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_readme_stays_a_compact_front_door -q
python - <<'PY'
from pathlib import Path
lines = [
    line
    for line in Path("README.md").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(len(lines))
assert len(lines) <= 60
PY
```

Expected: pytest passes and the line counter prints no more than `60`.

### Task 3: Align Contributor and Agent Guidance

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/contributing.md`

- [x] **Step 1: Update guidance**

Both docs should say README is a concise front door and detailed interface
examples/contracts belong in focused docs pages, not repeated in README.

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
PYTHONPATH=$PWD/src mkdocs build --strict
```

Expected: both commands exit `0`. The existing upstream Material for MkDocs
warning may appear.

- [x] **Step 3: Request local subagent code review**

Dispatch a local reviewer before opening the PR. Resolve any Critical or
Important findings before pushing.

## Verification

- RED:
  `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_readme_stays_a_compact_front_door -q`,
  `1 failed` because the previous README still had a standalone Python section.
- GREEN:
  `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py::test_readme_stays_a_compact_front_door -q`,
  `1 passed`; the README nonblank line counter printed `44` after package-safe
  links were added.
- Targeted metadata tests:
  `PYTHONPATH=$PWD/src pytest tests/test_package_metadata.py -q`, `8 passed`.
- Docs and formatting:
  `PYTHONPATH=$PWD/src mkdocs build --strict` passed with the known Material
  warning; `pre-commit run --all-files --show-diff-on-failure` passed; `git diff
  --check` passed.
- Local review follow-up:
  the first local review found that repository-relative README links were unsafe for
  package renderers because README is the PyPI long description. README links now
  use absolute ReadTheDocs/GitHub URLs, and the metadata guard rejects
  repository-relative `](docs/` links.
- Second local review follow-up:
  the reviewer found the pre-existing SkillCheck badge image returned `404`.
  The broken badge was removed, and the README guard now rejects `skillcheck`.
  Remaining README badge images returned `200` with a `GET` request.
- Hosted review follow-up:
  Gemini noted that a plain `](docs/` check would miss `./docs/` and `../docs/`
  links. The guard now uses a regex for relative docs links.
