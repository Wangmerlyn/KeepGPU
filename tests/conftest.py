import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-rocm",
        action="store_true",
        default=False,
        help="run tests marked as rocm (require ROCm stack)",
    )
    parser.addoption(
        "--run-macm",
        action="store_true",
        default=False,
        help="run tests marked as macm (require Apple Silicon MPS)",
    )
    parser.addoption(
        "--run-large-memory",
        action="store_true",
        default=False,
        help="run tests marked as large_memory (may allocate large VRAM)",
    )


def _skip_marked(items, marker_name, reason):
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker(marker_name) is not None:
            item.add_marker(skip_marker)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-rocm"):
        _skip_marked(items, "rocm", "need --run-rocm option to run")

    if not config.getoption("--run-macm"):
        _skip_marked(items, "macm", "need --run-macm option to run")

    if not config.getoption("--run-large-memory"):
        _skip_marked(
            items,
            "large_memory",
            "need --run-large-memory option to run",
        )


@pytest.fixture
def rocm_available():
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
    except Exception:
        return False


@pytest.fixture
def macm_available():
    try:
        import sys

        import torch

        return bool(sys.platform == "darwin" and torch.backends.mps.is_available())
    except Exception:
        return False
