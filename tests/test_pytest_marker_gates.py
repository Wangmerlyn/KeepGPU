from __future__ import annotations

from pathlib import Path

pytest_plugins = ("pytester",)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFTEST = PROJECT_ROOT / "tests" / "conftest.py"


def _install_project_conftest(pytester):
    pytester.makeconftest(PROJECT_CONFTEST.read_text(encoding="utf-8"))
    pytester.makeini(
        """
        [pytest]
        markers =
            large_memory: tests that use large VRAM
        """
    )


def test_large_memory_tests_skip_without_explicit_opt_in(pytester):
    _install_project_conftest(pytester)
    pytester.makepyfile(
        """
        import pytest


        @pytest.mark.large_memory
        def test_large_allocation_placeholder():
            raise AssertionError("large memory test should be opt-in")
        """
    )

    result = pytester.runpytest("-q", "-rs")

    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*need --run-large-memory option to run*"])


def test_large_memory_tests_run_with_explicit_opt_in(pytester):
    _install_project_conftest(pytester)
    pytester.makepyfile(
        """
        import pytest


        @pytest.mark.large_memory
        def test_large_allocation_placeholder():
            assert True
        """
    )

    result = pytester.runpytest("--run-large-memory", "-q")

    result.assert_outcomes(passed=1)
