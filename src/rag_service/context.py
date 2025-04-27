from pydantic import BaseModel
from typing import Optional



class Context(BaseModel):
    text: str
    document_name: str
    category: Optional[str] = None
    NPC: Optional[int] = None  # Kept for backward compatibility

