import pytest

from src.config import Config


def pytest_configure(config):
    # Initialize configuration before tests run
    backend_config = Config()  # This ensures .env is loaded

    # Check if RUNNING_TESTS is set to True
    if not backend_config.RUNNING_TESTS:
        pytest.exit("RUNNING_TESTS is not set to True. Exiting tests.")
