import random
import string

import pytest

from src.config import Config


def pytest_configure(config):
    # Initialize configuration before tests run
    backend_config = Config()  # This ensures .env is loaded

    # Set MongoDB database name to be random to avoid collisions
    # during parallel test runs
    random_suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=12)
    )
    backend_config.MONGODB_DATABASE = f"test_db_{random_suffix}"

    # Check if RUNNING_TESTS is set to True
    if not backend_config.RUNNING_TESTS:
        pytest.exit("RUNNING_TESTS is not set to True. Exiting tests.")
