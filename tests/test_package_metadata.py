import re
from pathlib import Path

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


def _runtime_dependency_name(dependency: str) -> str:
    match = re.match(r"^([a-zA-Z0-9._-]+)", dependency.strip())
    name = match.group(1) if match else ""
    return re.sub(r"[-_.]+", "-", name).lower()


def _runtime_dependency_names():
    return {
        _runtime_dependency_name(dependency) for dependency in _runtime_dependencies()
    }


def test_runtime_dependency_name_parser_handles_common_requirement_shapes():
    assert _runtime_dependency_name("rich>=13.8.0") == "rich"
    assert (
        _runtime_dependency_name("My_Package.Name[extra]>=1; python_version >= '3.9'")
        == "my-package-name"
    )
    assert _runtime_dependency_name("rich (>=13.8.0)") == "rich"
    assert (
        _runtime_dependency_name("pip @ git+https://github.com/pypa/pip.git") == "pip"
    )


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


def test_readme_stays_a_compact_front_door():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = readme.lower()
    lines = [line for line in readme.splitlines() if line.strip()]

    assert len(lines) <= 38
    assert readme.count("[![") <= 3
    assert "## choose an interface" not in normalized_readme
    assert "| need | start here |" not in normalized_readme
    assert "cuda" in normalized_readme
    assert "rocm/hip" in normalized_readme
    assert "mps" in normalized_readme
    assert "small, polite gpu keeper" in normalized_readme
    assert "## quick start" in normalized_readme
    assert "pip install keep-gpu" in normalized_readme
    assert re.search(r"^keep-gpu\s+--gpu-ids\s+0\b", readme, re.MULTILINE)
    assert "ctrl+c" in normalized_readme
    assert "## python" not in normalized_readme
    assert "## service, dashboard, and mcp" not in normalized_readme
    assert "### mcp and service api" not in normalized_readme
    assert "platform installs at a glance" not in normalized_readme
    assert "```bibtex" not in normalized_readme
    assert "skillcheck" not in normalized_readme
    assert not re.search(r"\]\(\.?\.?/?docs/", readme)
    assert "https://keepgpu.readthedocs.io/en/latest/" in readme
    assert "https://keepgpu.readthedocs.io/en/latest/getting-started/" in readme
    assert "https://keepgpu.readthedocs.io/en/latest/citation/" in readme


def test_public_docs_do_not_regress_to_cuda_only_or_experimental_mcp():
    public_docs = {
        "index": (PROJECT_ROOT / "docs/index.md").read_text(encoding="utf-8"),
        "getting_started": (PROJECT_ROOT / "docs/getting-started.md").read_text(
            encoding="utf-8"
        ),
        "architecture": (PROJECT_ROOT / "docs/concepts/architecture.md").read_text(
            encoding="utf-8"
        ),
        "contributing": (PROJECT_ROOT / "docs/contributing.md").read_text(
            encoding="utf-8"
        ),
    }

    assert "low-cost CUDA workloads" not in public_docs["index"]
    assert (
        "NVIDIA drivers + CUDA runtime visible to PyTorch"
        not in public_docs["getting_started"]
    )
    assert (
        "A non-zero integer indicates CUDA is available."
        not in public_docs["getting_started"]
    )
    assert "burst of CUDA ops" not in public_docs["architecture"]
    assert "[CudaGPUController rank=0]" not in public_docs["architecture"]
    assert "MCP server (experimental)" not in public_docs["contributing"]
    assert "RocmGPUController" in public_docs["index"]
    assert "MacMGPUController" in public_docs["index"]


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
