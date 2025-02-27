from pydantic import BaseModel
from typing import Optional

class ProgressData(BaseModel):
    taskName: str
    status: str  #"start", "complete"
    userId: Optional[str] = None
