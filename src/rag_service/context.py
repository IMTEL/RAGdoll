
from pydantic import BaseModel


class Context(BaseModel):
    text: str
    document_name: str
    category: str | None = None
    NPC: int | None = None  # Kept for backward compatibility
