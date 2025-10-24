import os

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv


# Read the key from the environment and normalize it
# Only load .env if FERNET_KEY is not already set (allows tests to override)
if not os.getenv("FERNET_KEY"):
    load_dotenv()  # take environment variables from .env file if present
fernet_key = os.getenv("FERNET_KEY")

if fernet_key is not None:
    # Strip surrounding whitespace
    fernet_key = fernet_key.strip()
    # Remove surrounding single quotes if present
    if fernet_key.startswith('"') and fernet_key.endswith('"'):
        fernet_key = fernet_key[1:-1].strip()

if not fernet_key:
    raise RuntimeError("Fernet Key is missing from .env")

try:
    # Attempt to construct a Fernet instance to validate the key
    encryptor = Fernet(fernet_key.encode("utf-8"))
    print("Fernet key validation: SUCCESS")
except Exception as e:
    print(f"Fernet key validation: FAILED - {e}")
    raise ValueError(
        "FERNET_KEY must be a valid 32-byte url-safe base64-encoded string"
    ) from e


def encrypt_str(api_key: str) -> str:
    """Encrypt a sensitive value (API key) and return the Fernet token as a UTF-8 string.

    Fernet.encrypt already returns a URL-safe base64-encoded token as bytes, so no
    additional base64 encoding is required.
    """
    if not api_key:
        raise ValueError("value must not be empty")
    token_bytes = encryptor.encrypt(api_key.encode("utf-8"))
    return token_bytes.decode("utf-8")


def decrypt_value(encrypted_value: str) -> str:
    """Decrypt a Fernet token (as a UTF-8 string) back into its readable form."""
    try:
        token_bytes = encrypted_value.encode("utf-8")
        return encryptor.decrypt(token_bytes).decode("utf-8")
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
