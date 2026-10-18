from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


def as_utc(value: datetime | None) -> datetime | None:
    """Returns ``value`` as an aware UTC datetime.

    Naive datetimes are taken to be UTC already: that is how MongoDB stores
    them and how it hands them back.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class AccessKeyPublic(BaseModel):
    """Access-key metadata that is safe to return from the API.

    Never contains the key, its hash or its encrypted copy. ``prefix`` is the
    first characters of the key, so users can tell their keys apart.
    ``view_once`` is False for keys created to be shown again later;
    ``revealable`` says whether that is still possible (an encrypted copy is
    stored), so the reveal endpoint can be offered.
    """

    id: str
    name: str
    prefix: str | None = None
    expiry_date: datetime | None = None
    created: datetime | None = None
    last_use: datetime | None = None
    view_once: bool = True
    revealable: bool = False

    @model_validator(mode="before")
    @classmethod
    def _derive_revealable(cls, data: Any) -> Any:
        # Built from a stored key's dump: derive the flag from the key itself
        # rather than trusting a stored copy of it.
        if isinstance(data, dict) and "key_encrypted" in data:
            data = {
                **data,
                "revealable": not data.get("view_once", True)
                and bool(data.get("key_encrypted")),
            }
        return data

    @field_validator("expiry_date", "created", "last_use")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        return as_utc(value)


class AccessKey(AccessKeyPublic):
    """Stored access key. Is used to authorize chatting with an agent.

    Keys are stored as a SHA-256 hash (``key_hash``). ``key`` only holds the
    plaintext of legacy keys created before hashing, until the
    ``scripts/migrate_hash_access_keys.py --drop-plaintext`` migration removes
    it. Keys created with ``view_once=False`` also keep ``key_encrypted``, the
    plaintext encrypted with FERNET_KEY, so the agent's members can reveal it
    again; authentication still only uses the hash. Never return this model
    from an endpoint; use ``AccessKeyPublic``.
    """

    key: str | None = None
    key_hash: str | None = None
    key_encrypted: str | None = None


class AccessKeyCreated(AccessKeyPublic):
    """Response for a newly created key: the only time the plaintext is shown."""

    key: str
