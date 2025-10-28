import bcrypt
from cryptography.fernet import Fernet, InvalidToken

from src.config import Config


ENCRYPTOR = Fernet(Config().FERNET_KEY.encode("utf-8"))


def encrypt_str(api_key: str) -> str:
    """Encrypt a sensitive value (API key) and return the Fernet token as a UTF-8 string.

    Fernet.encrypt already returns a URL-safe base64-encoded token as bytes, so no
    additional base64 encoding is required.
    """
    if not api_key:
        raise ValueError("value must not be empty")
    token_bytes = ENCRYPTOR.encrypt(api_key.encode("utf-8"))
    return token_bytes.decode("utf-8")


def decrypt_value(encrypted_value: str) -> str:
    """Decrypt a Fernet token (as a UTF-8 string) back into its readable form."""
    try:
        token_bytes = encrypted_value.encode("utf-8")
        return ENCRYPTOR.decrypt(token_bytes).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Invalid encryption token or wrong Fernet key") from e


def hash_access_key(access_key: str) -> bytes:
    """Hash an agent access key for secure storage."""
    if not access_key:
        raise ValueError("access_key must not be empty")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(access_key.encode("utf-8"), salt)


def verify_access_key(access_key: str, hashed: bytes) -> bool:
    """Verify an agent access key against its hash."""
    return bcrypt.checkpw(access_key.encode("utf-8"), hashed)
