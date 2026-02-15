import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-rocm",
        action="store_true",
        default=False,
        help="run tests marked as rocm (require ROCm stack)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-rocm"):
        return

    skip_rocm = pytest.mark.skip(reason="need --run-rocm option to run")
    for item in items:
        if "rocm" in item.keywords:
            item.add_marker(skip_rocm)


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
