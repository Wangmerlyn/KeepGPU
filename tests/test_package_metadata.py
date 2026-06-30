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
