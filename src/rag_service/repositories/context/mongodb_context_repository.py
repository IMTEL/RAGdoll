"""MongoDB implementation for context document storage."""

from pymongo import MongoClient

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.embeddings import similarity_search
from src.rag_service.repositories.context.base import ContextRepository


config = Config()


class MongoDBContextRepository(ContextRepository):
    """MongoDB-backed repository for document contexts with vector search.

    Uses MongoDB Atlas Vector Search for semantic similarity queries
    and text search for category-based retrieval.
    """

    def __init__(self):
        """Initialize MongoDB connection and set similarity threshold."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_CONTEXT_COLLECTION]
        self.similarity_threshold = 0.5

    def get_context_by_category(self, category: str) -> list[Context]:
        """Fetch all contexts for a specific category using Atlas Search.

        Args:
            category (str): The category to search for

        Returns:
            list[Context]: All contexts matching the category

        Raises:
            ValueError: If category is None/empty or no documents found
        """
        if not category:
            raise ValueError("Category cannot be empty")

        # Using MongoDB Atlas Search with an index for category field
        query = {
            "$search": {
                "index": "category",
                "text": {"path": "category", "query": category},
            }
        }

        # Execute the aggregate pipeline
        documents = self.collection.aggregate(
            [
                query,
                {"$limit": 50},  # Limit results to prevent overwhelming responses
            ]
        )

        # Convert cursor to a list
        documents = list(documents)
        if not documents:
            raise ValueError(f"No documents found for category: {category}")

        results = []
        for doc in documents:
            results.append(
                Context(
                    text=doc["text"],
                    document_name=doc["document_name"],
                    category=doc["category"],
                )
            )

        return results

    def get_context(self, document_id: str, embedding: list[float]) -> list[Context]:
        """Retrieve relevant contexts using vector similarity search.

        Uses MongoDB Atlas Vector Search to find the most semantically similar
        documents, then filters by similarity threshold.

        Args:
            document_id (str): Document identifier (for future use)
            embedding (list[float]): Query embedding vector

        Returns:
            list[Context]: Top matching contexts above similarity threshold

        Raises:
            ValueError: If embedding is None or no documents found
        """
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        query = {
            "$vectorSearch": {
                "index": "embeddings",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 30,
                "limit": 3,
            }
        }

        documents = self.collection.aggregate([query])

        if not documents:
            raise ValueError("No documents found")

        documents = list(documents)

        results = []

        # Filter by similarity threshold to ensure quality
        for document in documents:
            if (
                similarity_search(embedding, document["embedding"])
                > self.similarity_threshold
            ):
                results.append(
                    Context(
                        text=document["text"],
                        document_name=document["document_name"],
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
        """Store a new context document with metadata and embedding."""
        text = context.text
        document_name = context.document_name
        category = context.category

        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not category:
            raise ValueError("Category cannot be empty")
        if not embedding:
            raise ValueError("embedding cannot be empty")

        self.collection.insert_one(
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
        """Verify MongoDB connection health.

        Returns:
            bool: True if connection is active
        """
        try:
            self.client.admin.command("ping")
            print("Successfully pinged MongoDB")
            return True
        except Exception as e:
            print(f"Failed to ping MongoDB: {e}")
            return False
