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
    def get_context_by_category(self, category: str) -> list[Context]:
        """Fetch all contexts associated with the given category.

        Args:
            category (str): Document category to filter by

        Returns:
            list[Context]: List of contexts matching the category
        """

    @abstractmethod
    def get_context_by_corpus_ids(
        self,
        corpus_ids: list[str],
        embedding: list[float],
        num_candidates: int = 50,
        top_k: int = 5,
    ) -> list[Context]:
        """Retrieve contexts from specific corpus IDs using semantic similarity.

        Args:
            corpus_ids (list[str]): List of corpus/category identifiers to search within
            embedding (list[float]): Query embedding vector for similarity search
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return

        Returns:
            list[Context]: Most relevant contexts from the specified corpus
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
        embedding: list[float],
        context: Context,
    ) -> Context:
        """Store a new context document with its embedding.

        Args:
            document_id (str): Unique identifier for the document
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
