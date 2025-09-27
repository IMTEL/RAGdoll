from datetime import UTC, datetime

from pydantic import BaseModel, Field


class FailureData(BaseModel):
    errorCode: str
    description: str
    user_id: str | None = None  # Optional
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
