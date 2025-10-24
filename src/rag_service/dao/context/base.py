"""Abstract base classes for DAO pattern.

This module defines the interfaces for document context and agent DAOs.
"""

from abc import ABC, abstractmethod

from src.rag_service.context import Context


class ContextDAO(ABC):
    """Abstract base class for document context storage.

    This DAO handles storage and retrieval of document contexts
    with embeddings for semantic search.
    """

    @abstractmethod
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
        """Retrieve contexts for an agent using semantic similarity.

        Searches within the agent's documents, filtered by document IDs provided by roles.
        Uses vector similarity search with indexes on agent_id and document_id for efficiency.

        Args:
            agent_id (str): Agent identifier
            query_embedding (list[float]): Query embedding vector for similarity search
            query_text (str): Full query text for vector search (includes context)
            keyword_query_text (str | None): Simplified query text for BM25 keyword search.
                                              If None, uses query_text for both searches.
            documents (list[str] | None): Optional list of document IDs to filter by
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return
            similarity_threshold (float | None): Minimum similarity score for results.
                                                  If None, uses implementation default.
            hybrid_search_alpha (float | None): Weight for hybrid search (0=keyword, 1=vector).
                                                 If None, uses implementation default.

        Returns:
            list[Context]: Most relevant contexts for the agent
        """

    @abstractmethod
    def get_context(self, document_id: str, embedding: list[float]) -> list[Context]:
        """Retrieve context using semantic similarity search.

        Args:
            document_id (str): Identifier for the document
            embedding (list[float]): Query embedding vector for similarity search

        Returns:
            list[Context]: Most relevant contexts based on embedding similarity
        """

    @abstractmethod
    def insert_context(
        self,
        document_id: str,
        agent_id: str,
        embedding: list[float],
        context: Context,
    ) -> Context:
        """Store a new context document with its embedding.

        Args:
            document_id (str): Unique identifier for the document
            agent_id (str): Agent identifier that owns this document
            embedding (list[float]): Vector embedding of the text
            context (Context): The context object to store

        Returns:
            Context: The stored context object
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
