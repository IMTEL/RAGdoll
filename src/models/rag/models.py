"""RAG (Retrieval-Augmented Generation) domain models."""

from pydantic import BaseModel, Field


class RAGPostModel(BaseModel):
    """Data model for posting documents to the RAG system.

    Represents a document chunk with its embedding that will be
    stored in the vector database for semantic search.

    Attributes:
        text: The actual text content of the document chunk
        document_id: Unique identifier for the source document
        document_name: Human-readable name of the source document
        npc: NPC identifier associated with this content
        embedding: Vector embedding of the text
    """

    text: str
    document_id: str
    document_name: str
    npc: int
    embedding: list[float] = Field(default_factory=list)
