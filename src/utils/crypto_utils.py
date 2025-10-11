import os

from cryptography.fernet import Fernet, InvalidToken


# Read the key from the environment
fernet_key = os.getenv("FERNET_KEY")

if not fernet_key:
    raise RuntimeError("Fernet Key is missing from .env")

encryptor = Fernet(fernet_key.encode())


def encrypt_str(api_key: str) -> bytes:
    """Encrypt a sensitive value (API key) before storing it."""
    if api_key is None:
        raise ValueError("value must not be None")

    return encryptor.encrypt(api_key.encode())


def decrypt_value(encrypted_value: bytes) -> str:
    """Decrypt an encrypted value back into its readable form."""
    try:
        return encryptor.decrypt(encrypted_value).decode()
    except InvalidToken as e:
        raise ValueError("Invalid encryption token or wrong Fernet key") from e
