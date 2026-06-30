from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from setuptools.config.pyprojecttoml import read_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependency_lists():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["project"]["optional-dependencies"]


def _setuptools_config():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["tool"]["setuptools"]


def _project_config():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["project"]


def _build_system_requires():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["build-system"]["requires"]


def _runtime_dependencies():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["project"]["dependencies"]


def _runtime_dependency_names():
    return {
        canonicalize_name(Requirement(dependency).name)
        for dependency in _runtime_dependencies()
    }


def test_rocm_extra_is_declared_without_non_pypi_rocm_smi_dependency():
    extras = _optional_dependency_lists()

    assert "rocm" in extras
    assert "rocm-smi" not in extras["rocm"]


def test_colorlog_is_not_a_required_runtime_dependency():
    assert "colorlog" not in _runtime_dependency_names()


def test_license_metadata_uses_spdx_string_supported_by_python_floor():
    assert _project_config()["requires-python"] == ">=3.9"
    assert _project_config()["license"] == "MIT"
    assert "setuptools>=77.0.1" in _build_system_requires()


def test_readme_markdown_code_fences_are_balanced():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    fences = [line for line in readme.splitlines() if line.strip().startswith("```")]

    assert len(fences) % 2 == 0


def test_sdist_manifest_does_not_package_test_suite():
    manifest_path = PROJECT_ROOT / "MANIFEST.in"
    assert manifest_path.exists()
    manifest = manifest_path.read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert "prune tests" in active_lines
    assert not any(
        parts[0] in {"graft", "include", "recursive-include"}
        and len(parts) > 1
        and parts[1].lstrip("./").startswith("tests")
        for parts in (line.split() for line in active_lines)
    )


def test_package_data_only_includes_dashboard_static_assets():
    package_data = _setuptools_config()["package-data"]

    assert package_data == {
        "keep_gpu.mcp": [
            "static/index.html",
            "static/assets/*.css",
            "static/assets/*.js",
        ],
    }
