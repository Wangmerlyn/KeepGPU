import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQUIREMENTS_INSTALL_RE = re.compile(
    r"\b(?:python\s+-m\s+)?pip\s+install\s+-r\s+(?:\./)?requirements\.txt\b"
)


def _installs_root_requirements_file(workflow: str) -> bool:
    executable_lines = [
        line for line in workflow.splitlines() if not line.lstrip().startswith("#")
    ]
    return bool(ROOT_REQUIREMENTS_INSTALL_RE.search("\n".join(executable_lines)))


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
