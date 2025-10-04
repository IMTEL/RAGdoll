"""Mock implementation of ContextRepository for testing.

This module provides in-memory mock implementations that mimic the behavior
of real database repositories without requiring actual database connections.
"""

from src.rag_service.context import Context
from src.rag_service.repositories.base import ContextRepository


class MockContextRepository(ContextRepository):
    """In-memory singleton implementation of ContextRepository for testing.

    This mock repository stores all data in memory and persists across
    instances using the singleton pattern.
    """

    _instance = None  # Singleton instance
    _initialized = False  # Track initialization status

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the mock repository (only once)."""
        # Prevent reinitialization
        if not MockContextRepository._initialized:
            self.data = []  # In-memory storage
            self.similarity_threshold = 0.7
            self.collection = self  # For compatibility with code expecting .collection
            MockContextRepository._initialized = True

    def get_context_by_category(self, category: str) -> list[Context]:
        """Retrieve all contexts matching a category.

        Args:
            category (str): The category to filter by

        Returns:
            list[Context]: All contexts with matching category

        Raises:
            ValueError: If category is None or empty
        """
        if not category:
            raise ValueError("Category cannot be empty")

        results = []
        for document in self.data:
            if document.get("category") == category:
                results.append(
                    Context(
                        text=document["text"],
                        document_name=document["document_name"],
                        category=document["category"],
                    )
                )
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
                        category=document.get(
                            "category", f"npc_{document.get('npc', 'Unknown')}"
                        ),
                    )
                )
        return results

    def insert_context(
        self,
        document_id: str,
        embedding: list[float],
        context: Context,
    ) -> Context:
        """Store a new context in memory."""
        text = context.text
        document_name = context.document_name
        category = context.category

        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not category:
            raise ValueError("Category cannot be empty")
        if not embedding:
            raise ValueError("embedding cannot be empty")

        self.data.append(
            {
                "text": text,
                "document_name": document_name,
                "category": category,
                "embedding": embedding,
                "document_id": document_id,
            }
        )
        return context

    def is_reachable(self) -> bool:
        """Check if repository is reachable.

        Returns:
            bool: Always True for in-memory storage
        """
        return True

    def clear(self):
        """Clear all stored data. Useful for test cleanup."""
        self.data.clear()
