from pydantic import BaseModel
from typing import Optional



class Context(BaseModel):
    text: str
    document_name: str
    NPC: Optional[int] = None

