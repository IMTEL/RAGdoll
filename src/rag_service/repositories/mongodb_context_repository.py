"""MongoDB implementation for context document storage."""

from pymongo import MongoClient

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.embeddings import similarity_search
from src.rag_service.repositories.base import ContextRepository


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
        self.collection = self.db[config.MONGODB_COLLECTION]
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
            raise ValueError("Category cannot be None or empty")

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

    def get_context_from_npc(self, npc: int) -> list[Context]:
        """Legacy method for backward compatibility with NPC-based queries.

        Args:
            npc (int): The NPC identifier

        Returns:
            list[Context]: Contexts associated with the NPC

        Raises:
            ValueError: If npc is None or no documents found
        """
        if not npc:
            raise ValueError("NPC cannot be None")

        # Using MongoDB Atlas Search with an index named "npc"
        query = {"$search": {"index": "npc", "text": {"path": "npc", "query": npc}}}

        documents = self.collection.aggregate(
            [
                query,
                {"$limit": 50},
            ]
        )

        documents = list(documents)
        if not documents:
            raise ValueError(f"No documents found for NPC: {npc}")

        results = []
        for doc in documents:
            results.append(
                Context(
                    text=doc["text"],
                    document_name=doc["document_name"],
                    category=doc.get("category", f"npc_{doc['npc']}"),
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
            raise ValueError("Embedding cannot be None")

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

    def post_context(
        self,
        text: str,
        document_name: str,
        category: str,
        embedding: list[float],
        document_id: str,
    ) -> bool:
        """Store a new context document with metadata and embedding.

        Args:
            text (str): The text content
            document_name (str): Name of the source document
            category (str): Document category
            embedding (list[float]): Vector embedding
            document_id (str): Unique identifier

        Returns:
            bool: True if successfully stored

        Raises:
            ValueError: If any required field is None/empty
        """
        if not text:
            raise ValueError("text cannot be None")
        if not category:
            raise ValueError("Category cannot be None or empty")
        if not document_name:
            raise ValueError("Document name cannot be None")
        if not embedding:
            raise ValueError("Embedding cannot be None")

        try:
            self.collection.insert_one(
                {
                    "text": text,
                    "document_name": document_name,
                    "category": category,
                    "embedding": embedding,
                    "document_id": document_id,
                }
            )
            return True
        except Exception as e:
            print("Error in post_context:", e)
            return False

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
