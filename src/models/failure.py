from pydantic import BaseModel
from typing import Optional

class FailureData(BaseModel):
    errorCode: str
    description: str
    userId: Optional[str] = None  # Optional
