from pydantic import BaseModel, Field


class Context(BaseModel):
    """Context document for RAG retrieval.

    Attributes:
        text: The text content of the context
        document_name: Name of the source document
        document_id: Unique identifier for the document
        chunk_id: Unique identifier for this specific chunk (if chunked)
        chunk_index: Position of this chunk in the document (0-indexed)
        total_chunks: Total number of chunks the document was split into

    TODO: Implement text scraping/chunking for large documents.
    Currently storing whole documents. Future enhancement should:
    - Split documents into semantic text blocks/chunks
    - Store each chunk with proper metadata
    - Enable retrieval of relevant chunks rather than entire documents
    """

    text: str
    document_name: str
    document_id: str | None = Field(default=None)
    chunk_id: str | None = Field(default=None)
    chunk_index: int | None = Field(default=None)
    total_chunks: int | None = Field(default=1)
