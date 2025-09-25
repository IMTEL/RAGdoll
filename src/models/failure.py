from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone


class FailureData(BaseModel):
    errorCode: str
    description: str
    userId: Optional[str] = None  # Optional
    receivedAt: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
