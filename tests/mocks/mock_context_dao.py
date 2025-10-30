"""Mock implementation of ContextDAO for testing.

This module provides in-memory mock implementations that mimic the behavior
of a real DAO without requiring actual database connections.
"""

from src.rag_service.context import Context
from src.rag_service.dao import ContextDAO
from src.utils import singleton


@singleton
class MockContextDAO(ContextDAO):
    """In-memory singleton implementation of ContextDAO for testing.

    This mock DAO stores all data in memory and persists across
    instances using the singleton pattern.
    """

    def __init__(self):
        """Initialize the mock DAO."""
        self.data: list[dict] = []  # TODO: Change to list of Contexts
        self.similarity_threshold = 0.7
        self.collection = self  # For compatibility with code expecting .collection

    def get_context_for_agent(
        self,
        agent_id: str,
        query_embedding: list[float],
        query_text: str,
        keyword_query_text: str | None = None,
        documents: list[str] | None = None,
        num_candidates: int = 50,
        top_k: int = 5,
        similarity_threshold: float | None = None,
        hybrid_search_alpha: float | None = None,
    ) -> list[Context]:
        """Retrieve contexts for an agent using mock similarity.

        Args:
            agent_id (str): Agent identifier
            query_embedding (list[float]): Query embedding vector
            query_text (str): Full query text for vector search (includes context)
            keyword_query_text (str | None): Simplified query text for BM25 keyword search.
                                              If None, uses query_text for both searches.
            documents (list[str] | None): List of accessible documents
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return
            similarity_threshold (float | None): Minimum similarity score for results.
                                                  If None, uses instance default.
            hybrid_search_alpha (float | None): Weight for hybrid search (0=keyword, 1=vector).
                                                 If None, ignored in mock.

        Returns:
            list[Context]: Top matching contexts for the agent

        Raises:
            ValueError: If agent_id or embedding is empty
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not query_embedding:
            raise ValueError("Embedding cannot be empty")

        # Use provided threshold or fall back to instance default
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )

        results = []
        for document in self.data[:num_candidates]:  # Limit to num_candidates
            # Check if document belongs to agent
            if document.get("agent_id") != agent_id:
                continue

            # Filter by documents if provided
            if documents and document.get("_id") not in documents:
                continue

            # Mock similarity - returns high value for testing
            similarity = 0.9
            if similarity > threshold:
                doc_name = document.get("document_name", "default_document_name")
                results.append(
                    Context(
                        text=document["text"],
                        document_name=doc_name,
                        categories=document.get("categories", []),
                        document_id=document.get("document_id"),
                        chunk_id=document.get("chunk_id"),
                        chunk_index=document.get("chunk_index"),
                        total_chunks=document.get("total_chunks", 1),
                    )
                )
            # Limit results to top_k
            if len(results) >= top_k:
                break
        return results

    def get_context(self, document_id: str, embedding: list[float]) -> list[Context]:
        """Retrieve contexts using mock similarity search.

        Note: This mock implementation uses a fixed high similarity score
        for simplicity in testing.

        Args:
            document_id (str): Document identifier (not used in mock)
            embedding (list[float]): Query embedding vector

        Returns:
            list[Context]: All contexts above similarity threshold

        Raises:
            ValueError: If embedding is None
        """
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        results = []

        for document in self.data:
            # Mock similarity - returns high value for testing
            similarity = 0.9
            if similarity > self.similarity_threshold:
                doc_name = document.get("document_name", "default_document_name")
                results.append(
                    Context(
                        text=document["text"],
                        document_name=doc_name,
                        document_id=document.get("document_id"),
                        chunk_id=document.get("chunk_id"),
                        chunk_index=document.get("chunk_index"),
                        total_chunks=document.get("total_chunks", 1),
                    )
                )
        return results

    def insert_context(
        self,
        document_id: str,
        agent_id: str,
        embedding: list[float],
        context: Context,
    ) -> Context:
        """Store a new context in memory."""
        text = context.text
        document_name = context.document_name

        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not embedding:
            raise ValueError("embedding cannot be empty")

        document = {
            "text": text,
            "document_name": document_name,
            "embedding": embedding,
            "document_id": document_id,
            "agent_id": agent_id,
        }

        # Add optional chunking fields if present
        if context.chunk_id is not None:
            document["chunk_id"] = context.chunk_id
        if context.chunk_index is not None:
            document["chunk_index"] = context.chunk_index
        if context.total_chunks is not None:
            document["total_chunks"] = context.total_chunks

        self.data.append(document)
        return context

    def is_reachable(self) -> bool:
        """Check if DAO is reachable.

        Returns:
            bool: Always True for in-memory storage
        """
        return True

    # TODO: REMOVE - For MongoDB compatibility in context upload
    def delete_many(self, filter_dict: dict) -> object:
        """Delete multiple documents matching a filter (for compatibility with MongoDB).

        Args:
            filter_dict: Dictionary with filter criteria (e.g., {"document_id": "123"})

        Returns:
            Object with deleted_count attribute
        """

        class DeleteResult:
            def __init__(self, deleted_count: int):
                self.deleted_count = deleted_count

        initial_count = len(self.data)

        # Filter out documents that match the criteria
        if "document_id" in filter_dict:
            document_id = filter_dict["document_id"]
            self.data = [
                doc for doc in self.data if doc.get("document_id") != document_id
            ]

        deleted_count = initial_count - len(self.data)
        return DeleteResult(deleted_count)

    def clear(self):
        """Clear all stored data. Useful for test cleanup."""
        self.data.clear()
