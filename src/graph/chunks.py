from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import List

class Chunk(BaseModel):
    id: UUID
    document_id: UUID
    page_num: int
    chunk_index: int
    text: str
    embedding: List[float] = Field(..., min_items=1536, max_items=1536)
    created_at: datetime

    class Config:
        from_attributes = True