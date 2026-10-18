import importlib
import sys

import pytest
from cryptography.fernet import Fernet

from src.config import Config


MODULE = "src.utils.crypto_utils"


@pytest.fixture
def load_crypto(monkeypatch):
    """Returns a loader that imports a fresh crypto_utils for a FERNET_KEY value.

    The module reads the key at import time through the Config singleton, so
    both are reset. The original module and Config come back after the test,
    so other tests keep the app's real key.
    """
    monkeypatch.delitem(sys.modules, MODULE, raising=False)

    def load(fernet_key: str | None):
        if fernet_key is None:
            monkeypatch.delenv("FERNET_KEY", raising=False)
        else:
            monkeypatch.setenv("FERNET_KEY", fernet_key)
        Config._delete_instance__()
        sys.modules.pop(MODULE, None)
        return importlib.import_module(MODULE)

    yield load
    Config._delete_instance__()


def _new_key() -> str:
    return Fernet.generate_key().decode("utf-8")


class TestEncryptDecrypt:
    def test_roundtrip(self, load_crypto):
        crypto = load_crypto(_new_key())

        token = crypto.encrypt_str("my-super-secret")
        assert isinstance(token, str)
        assert token != "my-super-secret"
        assert crypto.decrypt_value(token) == "my-super-secret"

    def test_encrypt_empty_value_raises(self, load_crypto):
        crypto = load_crypto(_new_key())
        with pytest.raises(ValueError):
            crypto.encrypt_str("")

    def test_decrypt_invalid_token_raises(self, load_crypto):
        crypto = load_crypto(_new_key())
        with pytest.raises(ValueError):
            crypto.decrypt_value("not-a-token")

    def test_decrypt_token_from_another_key_raises(self, load_crypto):
        crypto = load_crypto(_new_key())
        foreign = Fernet(_new_key()).encrypt(b"secret").decode()
        with pytest.raises(ValueError):
            crypto.decrypt_value(foreign)


class TestKeyRotation:
    def test_first_key_encrypts_all_keys_decrypt(self, load_crypto):
        new_key, old_key = _new_key(), _new_key()
        old_token = Fernet(old_key).encrypt(b"old-secret").decode()

        crypto = load_crypto(f"{new_key},{old_key}")

        assert crypto.decrypt_value(old_token) == "old-secret"
        token = crypto.encrypt_str("new-secret")
        assert Fernet(new_key).decrypt(token.encode()) == b"new-secret"

    def test_whitespace_around_keys_is_ignored(self, load_crypto):
        new_key, old_key = _new_key(), _new_key()
        old_token = Fernet(old_key).encrypt(b"old-secret").decode()

        crypto = load_crypto(f" {new_key} , {old_key} ,")

        assert crypto.decrypt_value(old_token) == "old-secret"

    def test_rotate_value_reencrypts_with_the_first_key(self, load_crypto):
        new_key, old_key = _new_key(), _new_key()
        old_token = Fernet(old_key).encrypt(b"secret").decode()

        crypto = load_crypto(f"{new_key},{old_key}")
        rotated = crypto.rotate_value(old_token)

        assert Fernet(new_key).decrypt(rotated.encode()) == b"secret"

    def test_single_key_still_works(self, load_crypto):
        key = _new_key()
        token = Fernet(key).encrypt(b"secret").decode()
        crypto = load_crypto(key)
        assert crypto.decrypt_value(token) == "secret"


class TestConfiguration:
    def test_missing_fernet_key(self, load_crypto):
        with pytest.raises(RuntimeError):
            load_crypto(None)

    def test_invalid_fernet_key_raises_on_import(self, load_crypto):
        with pytest.raises(ValueError):
            load_crypto("not-a-valid-key")

    def test_one_invalid_key_in_the_list_raises_on_import(self, load_crypto):
        with pytest.raises(ValueError):
            load_crypto(f"{_new_key()},not-a-valid-key")

    def test_invalid_entry_error_names_position_not_value(self, load_crypto):
        bad = "not-a-valid-key-SENTINEL"
        with pytest.raises(ValueError) as excinfo:
            load_crypto(f"{_new_key()},{bad}")
        message = str(excinfo.value)
        assert "FERNET_KEY" in message
        assert "entry 2" in message
        assert "SENTINEL" not in message
        assert excinfo.value.__cause__ is None
