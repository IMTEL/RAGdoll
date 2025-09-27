
from pydantic import BaseModel


class RAGPostModel(BaseModel):
    text: str
    document_id: str
    document_name: str
    NPC: int
    embedding: list[float]
