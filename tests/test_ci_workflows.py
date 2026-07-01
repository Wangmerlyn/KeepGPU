import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQUIREMENTS_INSTALL_RE = re.compile(
    r"\b(?:python\s+-m\s+)?pip\s+install\s+-r\s+(?:\./)?requirements\.txt\b"
)
MKDOCS_MAJOR_TWO_BOUND_RE = re.compile(r"<\s*2(?:\.0+)?(?:\s|,|$)")
YAML_BLOCK_SCALAR_START_RE = re.compile(r":\s*[|>][-+0-9]*\s*$")
PYTHON_VERSION_MATRIX_RE = re.compile(r"python-version:\s*\[([^\]]+)\]")


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
    action_pattern = re.escape(action)
    block_scalar_indent = None
    for line in workflow.splitlines():
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        if block_scalar_indent is not None:
            if indent > block_scalar_indent:
                continue
            block_scalar_indent = None

        active_line = line.split("#", 1)[0]
        if YAML_BLOCK_SCALAR_START_RE.search(active_line.rstrip()):
            block_scalar_indent = indent
            continue

        active_line = active_line.strip()
        match = re.match(
            rf"-?\s*uses\s*:\s*['\"]?({action_pattern}@[^\s'\"]+)['\"]?\s*$",
            active_line,
        )
        if match is not None:
            return match.group(1)
    raise AssertionError(f"missing workflow action: {action}")


def _python_version_matrix(workflow: str) -> list[str]:
    match = PYTHON_VERSION_MATRIX_RE.search(workflow)
    if match is None:
        raise AssertionError("missing python-version matrix")
    return [
        version.strip().strip("\"'")
        for version in match.group(1).split(",")
        if version.strip()
    ]


def _normalized_pip_install_commands(workflow: str) -> list[str]:
    pip_install_commands = re.findall(
        r"^\s*(?:python\s+-m\s+)?pip\s+install\b.*$", workflow, re.MULTILINE
    )
    return [re.sub(r"\s+", " ", command.strip()) for command in pip_install_commands]


def test_precommit_workflow_does_not_install_runtime_package():
    workflow_path = PROJECT_ROOT / ".github/workflows/pre-commit.yaml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/pre-commit.yml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert _normalized_pip_install_commands(workflow) == [
        "python -m pip install --upgrade pip",
        "pip install pre-commit",
    ]


def test_precommit_workflow_uses_current_core_action_refs():
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

    assert (
        _workflow_action_ref(precommit_workflow, "actions/checkout")
        == expected_checkout
    )
    assert (
        _workflow_action_ref(precommit_workflow, "actions/setup-python")
        == expected_setup_python
    )


def test_workflow_action_ref_ignores_comments_and_allows_spacing():
    workflow = """
    steps:
      - name: stale note
        # uses: actions/checkout@v99
      - uses:    actions/checkout@v4
    """

    assert _workflow_action_ref(workflow, "actions/checkout") == "actions/checkout@v4"


def test_workflow_action_ref_requires_active_uses_line():
    workflow = """
    steps:
      - name: stale note
        # uses: actions/setup-python@v5
    """

    with pytest.raises(
        AssertionError, match="missing workflow action: actions/setup-python"
    ):
        _workflow_action_ref(workflow, "actions/setup-python")


def test_workflow_action_ref_ignores_run_block_content():
    workflow = """
    steps:
      - name: show historical action
        run: |
          uses: actions/checkout@v99
      - uses: actions/checkout@v4
    """

    assert _workflow_action_ref(workflow, "actions/checkout") == "actions/checkout@v4"


def test_python_workflow_does_not_install_root_requirements_file():
    workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yaml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert not _installs_root_requirements_file(workflow)


def test_python_workflow_tests_supported_floor_and_latest_advertised_version():
    workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yaml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert _python_version_matrix(workflow) == ["3.9", "3.13"]


def test_python_workflow_restores_metadata_setuptools_after_torch_install():
    workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yml"
    if not workflow_path.exists():
        workflow_path = PROJECT_ROOT / ".github/workflows/python-app.yaml"
    workflow = workflow_path.read_text(encoding="utf-8")

    assert _normalized_pip_install_commands(workflow) == [
        "python -m pip install --upgrade pip",
        "pip install pytest",
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        'pip install "setuptools>=77,<82"',
        "pip install -e .",
    ]


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
