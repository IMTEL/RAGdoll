from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.db.graph.enums import NodeType, EdgeType
from src.utils.canon import canon

class Node(BaseModel):
    id:   str
    type: str
    title: str
    chunk_ids: list[str] = Field(default_factory=list)
    properties: dict[str, str] = {}

    @field_validator("id", mode="before")
    def _canon_id(cls, v, values):           # slugify at model level
        return canon(v or values.get("title", ""))
    
    # uncomment when valid node types are able to be updated as you go
    # @field_validator("type", mode="before")
    # def _check_type(cls, v):
    #     v_up = v.strip().upper()
    #     if v_up not in NodeType.__members__:
    #         raise ValueError(f"Invalid node type: {v_up}")
    #     return v_up

class Edge(BaseModel):
    from_: str = Field(..., alias="from")
    to:    str
    type:  str
    chunk_ids: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("from_", "to", mode="before")
    def _canon_endpoints(cls, v):
        return canon(v)

    # @field_validator("type", mode="before")
    # def _check_edge_type(cls, v):
    #     v_up = v.strip().upper()
    #     if v_up not in EdgeType.__members__:
    #         raise ValueError(f"Invalid edge type: {v_up}")
    #     return v_up

class GraphPayload(BaseModel):
    graph_id: str
    chunk_id: str
    document_id: str
    nodes: list[Node]
    edges: list[Edge]


class GraphStructure(BaseModel):
    """Internal model for holding graph structure from Neo4j."""
    graph_id: str
    nodes: list[Node]
    edges: list[Edge]