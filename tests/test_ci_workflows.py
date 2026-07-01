import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQUIREMENTS_INSTALL_RE = re.compile(
    r"\b(?:python\s+-m\s+)?pip\s+install\s+-r\s+(?:\./)?requirements\.txt\b"
)
MKDOCS_MAJOR_TWO_BOUND_RE = re.compile(r"<\s*2(?:\.0+)?(?:\s|,|$)")


def _installs_root_requirements_file(workflow: str) -> bool:
    executable_lines = [
        line for line in workflow.splitlines() if not line.lstrip().startswith("#")
    ]
    return bool(ROOT_REQUIREMENTS_INSTALL_RE.search("\n".join(executable_lines)))


def _active_requirement_names(requirements: str) -> list[str]:
    return [
        re.split(r"\s*(?:[<>=!~;\\[]|$)", line.split("#", 1)[0].strip(), maxsplit=1)[0]
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    ]


def _active_requirement_line(requirements: str, name: str) -> str:
    for line in requirements.splitlines():
        requirement = line.split("#", 1)[0].strip()
        if not requirement:
            continue
        requirement_name = re.split(r"\s*(?:[<>=!~;\\[]|$)", requirement, maxsplit=1)[0]
        if requirement_name == name:
            return requirement
    raise AssertionError(f"missing requirement: {name}")


def _has_active_pymdown_extension(mkdocs_config: str) -> bool:
    return any(
        "pymdownx." in line.split("#", 1)[0] for line in mkdocs_config.splitlines()
    )


def _bounds_below_mkdocs_two(requirement: str) -> bool:
    return bool(MKDOCS_MAJOR_TWO_BOUND_RE.search(requirement))


def _workflow_action_ref(workflow: str, action: str) -> str:
    match = re.search(rf"uses:\s*{re.escape(action)}@v[0-9]+\b", workflow)
    assert match is not None, f"missing workflow action: {action}"
    return match.group(0).removeprefix("uses:").strip()


def test_precommit_workflow_does_not_install_runtime_package():
    workflow_path = PROJECT_ROOT / ".github/workflows/pre-commit.yaml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/pre-commit.yml"
    workflow = workflow_path.read_text(encoding="utf-8")
    pip_install_commands = re.findall(
        r"^\s*(?:python\s+-m\s+)?pip\s+install\b.*$", workflow, re.MULTILINE
    )
    normalized_commands = [
        re.sub(r"\s+", " ", command.strip()) for command in pip_install_commands
    ]

    assert normalized_commands == [
        "python -m pip install --upgrade pip",
        "pip install pre-commit",
    ]


def test_precommit_workflow_uses_current_core_action_majors():
    precommit_path = PROJECT_ROOT / ".github/workflows/pre-commit.yaml"
    if not precommit_path.exists():
        precommit_path = PROJECT_ROOT / ".github/workflows/pre-commit.yml"
    precommit_workflow = precommit_path.read_text(encoding="utf-8")

    python_workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yml"
    if not python_workflow_path.exists():
        python_workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yaml"
    python_workflow = python_workflow_path.read_text(encoding="utf-8")

    expected_checkout = _workflow_action_ref(python_workflow, "actions/checkout")
    expected_setup_python = _workflow_action_ref(
        python_workflow, "actions/setup-python"
    )

    assert f"uses: {expected_checkout}" in precommit_workflow
    assert f"uses: {expected_setup_python}" in precommit_workflow


def test_python_workflow_does_not_install_root_requirements_file():
    workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yaml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert not _installs_root_requirements_file(workflow)


def test_root_requirements_guard_allows_harmless_references():
    assert _installs_root_requirements_file("pip install -r requirements.txt")
    assert _installs_root_requirements_file(
        "python -m pip install -r ./requirements.txt"
    )
    assert not _installs_root_requirements_file(
        "# historical note: pip install -r requirements.txt is forbidden"
    )
    assert not _installs_root_requirements_file("pip install -r docs/requirements.txt")


def test_docs_requirements_include_direct_mkdocs_dependency():
    docs_requirements = (PROJECT_ROOT / "docs/requirements.txt").read_text(
        encoding="utf-8"
    )

    assert "mkdocs" in _active_requirement_names(docs_requirements)


def test_docs_requirements_bound_known_incompatible_mkdocs_major():
    docs_requirements = (PROJECT_ROOT / "docs/requirements.txt").read_text(
        encoding="utf-8"
    )

    assert _bounds_below_mkdocs_two(
        _active_requirement_line(docs_requirements, "mkdocs")
    )


def test_mkdocs_major_bound_guard_rejects_similar_but_unsafe_bounds():
    assert _bounds_below_mkdocs_two("mkdocs<2")
    assert _bounds_below_mkdocs_two("mkdocs >=1.6,<2.0")
    assert not _bounds_below_mkdocs_two("mkdocs<20")
    assert not _bounds_below_mkdocs_two("mkdocs<2.1")


def test_docs_requirements_include_configured_pymdown_extensions():
    mkdocs_config = (PROJECT_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    docs_requirements = (PROJECT_ROOT / "docs/requirements.txt").read_text(
        encoding="utf-8"
    )

    if _has_active_pymdown_extension(mkdocs_config):
        assert "pymdown-extensions" in _active_requirement_names(docs_requirements)


def test_docs_requirements_guard_ignores_commented_pymdown_extensions():
    assert not _has_active_pymdown_extension("# - pymdownx.tabbed\n")
    assert not _has_active_pymdown_extension("  - admonition # - pymdownx.tabbed\n")
