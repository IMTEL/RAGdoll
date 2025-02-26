from pydantic import BaseModel

class FailureData(BaseModel):
    errorCode: str
    description: str
    userId: str = None  # Optional
