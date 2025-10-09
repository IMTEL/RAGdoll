"""Error tracking domain models."""

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class FailureData(BaseModel):
    """Error/failure event tracking.

    Captures information about errors or failures that occur
    during system operation for monitoring and debugging.

    Attributes:
        error_code: Machine-readable error identifier
        description: Human-readable error explanation
        user_id: Optional user associated with the error
        received_at: When the error was recorded
    """

    error_code: str
    description: str
    user_id: str | None = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
