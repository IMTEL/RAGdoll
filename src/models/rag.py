from pydantic import BaseModel
from typing import List


class RAGPostModel(BaseModel):
    text: str
    document_id: str
    document_name: str
    NPC: int
    embedding: List[float]
