from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from src.config import Config


def _build_encryptor(fernet_keys: str) -> MultiFernet:
    """Builds the encryptor from FERNET_KEY, a comma-separated list of keys.

    The first key encrypts; every key decrypts. To rotate, put the new key
    first and keep the old one after it until every value is re-encrypted.
    """
    keys = [key.strip() for key in fernet_keys.split(",") if key.strip()]
    if not keys:
        raise ValueError("FERNET_KEY must contain at least one key")
    fernets = []
    for position, key in enumerate(keys, start=1):
        try:
            fernets.append(Fernet(key.encode("utf-8")))
        except ValueError:
            # Never echo the value: it is (or is close to) a secret.
            raise ValueError(
                f"FERNET_KEY entry {position} of {len(keys)} is not a valid Fernet "
                "key (expected 32 url-safe base64-encoded bytes)"
            ) from None
    return MultiFernet(fernets)


ENCRYPTOR = _build_encryptor(Config().FERNET_KEY)


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


def rotate_value(encrypted_value: str) -> str:
    """Re-encrypt a Fernet token with the first (current) FERNET_KEY."""
    try:
        return ENCRYPTOR.rotate(encrypted_value.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Invalid encryption token or wrong Fernet key") from e
