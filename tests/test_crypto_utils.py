import importlib
import os
import sys

import pytest

from src.config import Config


# Ensure required third-party packages are available; skip the test module if not.
pytest.importorskip("cryptography")
pytest.importorskip("bcrypt")

from cryptography.fernet import Fernet


def reload_crypto(monkeypatch, key: bytes | None):
    """Set or unset FERNET_KEY and reload the crypto_utils module.

    Behavior:
    - If `key` is provided (bytes), set it into env and import the module.
    - If `key` is None, prefer the existing `FERNET_KEY` from the environment.
      If the environment doesn't have one, skip the tests (we assume the
      project's .env provides the FERNET_KEY).
    """
    if key is None:
        # If caller doesn't provide a key, generate a fresh transient key
        # so unit tests exercise encryption in isolation and don't depend
        # on the developer's environment.
        gen = Fernet.generate_key()
        monkeypatch.setenv("FERNET_KEY", gen.decode("utf-8"))
    else:
        # ensure we pass a string value to the environment
        monkeypatch.setenv("FERNET_KEY", key.decode("utf-8"))

    # remove from sys.modules so importlib re-executes module-level code
    if "src.utils.crypto_utils" in sys.modules:
        del sys.modules["src.utils.crypto_utils"]

    return importlib.import_module("src.utils.crypto_utils")


class TestCryptoUtils:
    """Tests for src.utils.crypto_utils functions."""

    def test_module_import_with_valid_key(self, monkeypatch):
        # Generate a fresh key for isolation
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)
        # Basic attributes exist
        assert hasattr(crypto, "encrypt_str")
        assert hasattr(crypto, "decrypt_value")

    def test_encrypt_decrypt_roundtrip(self, monkeypatch):
        # Use a fresh ephemeral key for this test
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)

        secret = "my-super-secret"
        token = crypto.encrypt_str(secret)
        assert isinstance(token, str)
        assert token != secret

        decrypted = crypto.decrypt_value(token)
        assert decrypted == secret

    def test_encrypt_empty_value_raises(self, monkeypatch):
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)

        with pytest.raises(ValueError):
            crypto.encrypt_str("")

    def test_decrypt_invalid_token_raises(self, monkeypatch):
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)

        with pytest.raises(ValueError):
            crypto.decrypt_value("not-a-token")

    def test_hash_and_verify_access_key(self, monkeypatch):
        # Use a fresh ephemeral key for this test (module import requires a key)
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)

        access_key = "agent-key-123"
        hashed = crypto.hash_access_key(access_key)
        assert isinstance(hashed, (bytes, bytearray))

        assert crypto.verify_access_key(access_key, hashed) is True
        assert crypto.verify_access_key("wrong-key", hashed) is False

    def test_hash_empty_raises(self, monkeypatch):
        key = Fernet.generate_key()
        crypto = reload_crypto(monkeypatch, key)

        with pytest.raises(ValueError):
            crypto.hash_access_key("")

    def test_missing_fernet_key(self, monkeypatch):
        # Ensure env is unset and re-importing the module raises
        monkeypatch.delenv("FERNET_KEY", raising=False)

        # Clear config singleton to force re-read of env var
        Config._delete_instance__()

        with pytest.raises(RuntimeError):
            Config()

    def test_invalid_fernet_key_raises_on_import(self, monkeypatch):
        monkeypatch.setenv("FERNET_KEY", "not-a-valid-key")
        if "src.utils.crypto_utils" in sys.modules:
            del sys.modules["src.utils.crypto_utils"]

        with pytest.raises(ValueError):
            importlib.import_module("src.utils.crypto_utils")

    def test_reads_fernet_key_from_env(self, monkeypatch):
        """Test.

        If the environment already provides a FERNET_KEY (e.g. from .env),
        ensure the module uses it. This test will be skipped when the env var
        isn't present because it's specifically validating reading from the
        environment/.env file.
        """
        # Prefer an already-exported FERNET_KEY. If it's not present, try to
        # read only the FERNET_KEY value from the project's .env file (do not
        # source or export the whole file). This keeps the key out of the shell
        # environment while still allowing the test to validate env-based loading.
        existing = os.environ.get("FERNET_KEY")
        key_value = existing
        if not key_value:
            # locate the project root (one level up from tests/)
            tests_dir = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(tests_dir, os.pardir))
            env_path = os.path.join(project_root, ".env")
            if os.path.exists(env_path):
                # Parse the file for a single FERNET_KEY entry only (no execution,
                # no other variables loaded).
                with open(env_path, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "FERNET_KEY" in line and "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip() == "FERNET_KEY":
                                # strip optional surrounding quotes
                                val = v.strip()
                                if val.startswith('"') and val.endswith('"'):
                                    val = val[1:-1]
                                key_value = val
                                break

        if not key_value:
            pytest.skip(
                "FERNET_KEY not present in environment or .env; skipping env-read test"
            )

        # Ensure the module will be imported fresh and will pick up the env value
        # We set it using monkeypatch so the shell environment is not modified.
        monkeypatch.setenv("FERNET_KEY", key_value)
        if "src.utils.crypto_utils" in sys.modules:
            del sys.modules["src.utils.crypto_utils"]

        crypto = importlib.import_module("src.utils.crypto_utils")
        secret = "env-based-secret"
        token = crypto.encrypt_str(secret)
        assert crypto.decrypt_value(token) == secret
