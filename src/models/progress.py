from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

class ProgressData(BaseModel):
    taskName: str
    status: str  #"start", "complete"
    userId: Optional[str] = None
    startedAt: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))  # Task started
    completedAt: Optional[datetime] = None #Task completed
