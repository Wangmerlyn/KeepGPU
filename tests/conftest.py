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


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-rocm"):
        skip_rocm = pytest.mark.skip(reason="need --run-rocm option to run")
        for item in items:
            if "rocm" in item.keywords:
                item.add_marker(skip_rocm)

    if not config.getoption("--run-macm"):
        skip_macm = pytest.mark.skip(reason="need --run-macm option to run")
        for item in items:
            if "macm" in item.keywords:
                item.add_marker(skip_macm)


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
