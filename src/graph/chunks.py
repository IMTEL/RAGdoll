from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: UUID
    document_id: UUID
    page_num: int
    chunk_index: int
    text: str
    embedding: list[float] = Field(..., min_items=1536, max_items=1536)
    created_at: datetime

    class Config:
        from_attributes = True
