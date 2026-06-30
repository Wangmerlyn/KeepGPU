import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files():
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", "*.py"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    for relative_path in result.stdout.splitlines():
        yield REPO_ROOT / relative_path


def _has_future_annotations(tree: ast.Module) -> bool:
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
        ):
            if isinstance(node.value.value, str):
                continue
        if not isinstance(node, ast.ImportFrom) or node.module != "__future__":
            return False
        if any(alias.name == "annotations" for alias in node.names):
            return True
    return False


def _annotation_nodes(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.arg):
            if node.annotation is not None:
                yield node.annotation
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _uses_pep604_annotation(tree: ast.Module) -> bool:
    for annotation in _annotation_nodes(tree):
        if _contains_pep604_union(annotation):
            return True
    return False


def _contains_pep604_union(node: ast.AST) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return True
    if isinstance(node, ast.Subscript) and _is_annotated_name(node.value):
        args = _subscript_args(node.slice)
        if not args:
            return False
        return _contains_pep604_union(args[0])
    return any(_contains_pep604_union(child) for child in ast.iter_child_nodes(node))


def _is_annotated_name(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "Annotated"
    if isinstance(node, ast.Attribute):
        return node.attr == "Annotated"
    return False


def _subscript_args(node: ast.AST):
    if isinstance(node, ast.Index):
        node = node.value
    if isinstance(node, ast.Tuple):
        return list(node.elts)
    return [node]


def test_python_files_include_tracked_python_outside_src_and_tests():
    paths = {path.relative_to(REPO_ROOT) for path in _python_files()}

    assert Path("docs/__init__.py") in paths


def test_future_annotations_detection_allows_other_future_imports_first():
    tree = ast.parse(
        '"""module docstring"""\n'
        "from __future__ import generator_stop\n"
        "from __future__ import annotations\n"
        "\n"
        "def parse(value: str | None) -> str | None:\n"
        "    return value\n"
    )

    assert _has_future_annotations(tree)


def test_future_annotations_detection_rejects_misplaced_future_import():
    tree = ast.parse(
        "ready = True\n"
        "from __future__ import annotations\n"
        "\n"
        "def parse(value: str | None) -> str | None:\n"
        "    return value\n"
    )

    assert not _has_future_annotations(tree)


def test_annotation_nodes_include_all_function_argument_forms():
    tree = ast.parse(
        "def parse(\n"
        "    pos: str | None,\n"
        "    /,\n"
        "    normal: int | None,\n"
        "    *items: float | None,\n"
        "    kw: bool | None,\n"
        "    **extras: bytes | None,\n"
        ") -> str | None:\n"
        "    value: dict[str, str] | None = None\n"
        "    return str(value)\n"
    )

    pep604_annotations = 0
    for annotation in _annotation_nodes(tree):
        if any(
            isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
            for node in ast.walk(annotation)
        ):
            pep604_annotations += 1

    assert pep604_annotations == 7


def test_pep604_detector_ignores_annotated_metadata_bitor():
    tree = ast.parse(
        "from typing import Annotated\n"
        "\n"
        "class Flags:\n"
        "    A = 1\n"
        "    B = 2\n"
        "\n"
        "value: Annotated[int, Flags.A | Flags.B]\n"
    )

    assert not _uses_pep604_annotation(tree)


def test_pep604_detector_counts_annotated_type_union():
    tree = ast.parse(
        "from typing import Annotated\n"
        "\n"
        "value: Annotated[int | str, 'metadata']\n"
    )

    assert _uses_pep604_annotation(tree)


def test_pep604_annotations_are_postponed_for_py38_target():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        if not _uses_pep604_annotation(tree):
            continue
        if _has_future_annotations(tree):
            continue
        offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == [], "PEP 604 annotations need future annotations: " + ", ".join(
        offenders
    )
