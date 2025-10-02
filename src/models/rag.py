from pydantic import BaseModel


class RAGPostModel(BaseModel):
    text: str
    document_id: str
    document_name: str
    npc: int
    embedding: list[float]
