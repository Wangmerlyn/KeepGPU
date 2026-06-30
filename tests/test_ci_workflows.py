import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
