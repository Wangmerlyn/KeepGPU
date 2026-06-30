from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _optional_dependency_lists():
    config = read_configuration(
        str(PROJECT_ROOT / "pyproject.toml"),
        expand=False,
    )
    return config["project"]["optional-dependencies"]


def test_rocm_extra_is_declared_without_non_pypi_rocm_smi_dependency():
    extras = _optional_dependency_lists()

    assert "rocm" in extras
    assert "rocm-smi" not in extras["rocm"]
