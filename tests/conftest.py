import pytest

from src.config import Config


def pytest_configure(config):
    # Check if RUNNING_TESTS is set to True
    if not Config().RUNNING_TESTS:
        pytest.exit("RUNNING_TESTS is not set to True. Exiting tests.")
