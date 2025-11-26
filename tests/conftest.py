import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--run-rocm",
        action="store_true",
        default=False,
        help="run tests marked as rocm (require ROCm stack)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "rocm: tests that require ROCm stack")
    config.addinivalue_line("markers", "large_memory: tests that use large VRAM")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-rocm"):
        return

    skip_rocm = pytest.mark.skip(reason="need --run-rocm option to run")
    for item in items:
        if "rocm" in item.keywords:
            item.add_marker(skip_rocm)


@pytest.fixture
def rocm_available():
    return bool(torch.cuda.is_available() and getattr(torch.version, "hip", None))
