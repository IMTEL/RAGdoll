from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.utils.crypto_utils import decrypt_value


class UserAPIKey(BaseModel):
    """Stored API key metadata for a user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    provider: str
    usage: Literal["llm", "embedding", "both"]
    key_encrypted: str
    redacted_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_response(self) -> UserAPIKeyResponse:
        return UserAPIKeyResponse(
            id=self.id,
            label=self.label,
            provider=self.provider,
            usage=self.usage,
            redacted_key=self.redacted_key,
            created_at=self.created_at,
        )

    def to_detail(self) -> UserAPIKeyDetailResponse:
        return UserAPIKeyDetailResponse(
            id=self.id,
            label=self.label,
            provider=self.provider,
            usage=self.usage,
            redacted_key=self.redacted_key,
            created_at=self.created_at,
            raw_key=decrypt_value(self.key_encrypted),
        )


class UserAPIKeyResponse(BaseModel):
    id: str
    label: str
    provider: str
    usage: Literal["llm", "embedding", "both"]
    redacted_key: str
    created_at: datetime


class UserAPIKeyDetailResponse(UserAPIKeyResponse):
    raw_key: str
