from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional

class Node(BaseModel):
    id: str
    type: str
    title: str
    chunk_ids: list[str] = Field(default_factory=list)
    properties: dict[str, str | int | float | bool] = Field(default_factory=dict)

class Edge(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    type: str
    chunk_ids: list[str] = Field(default_factory=list)

    model_config = {
        "populate_by_name": True
    }

class Graph(BaseModel):
    # Metadata from Postgres
    id: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime

    # Structure from Neo4j
    nodes: list[Node]
    edges: list[Edge]

    model_config = {
        "from_attributes": True
    }