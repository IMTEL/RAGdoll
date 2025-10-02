from abc import ABC, abstractmethod

from pymongo import MongoClient

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.embeddings import similarity_search


config = Config()


class Database(ABC):
    """Abstract class for Connecting to a Database."""

    @classmethod
    def __instancecheck__(cls, instance: any) -> bool:
        return cls.__subclasscheck__(type(instance))

    @classmethod
    def __subclasscheck__(cls, subclass: any) -> bool:
        return (
            hasattr(subclass, "get_context") and callable(subclass.get_context)
        ) and (hasattr(subclass, "post_context") and callable(subclass.post_context))

    @abstractmethod
    def get_context_by_category(self, category: str) -> list[Context]:
        """Fetches context solely based on what context associated with the given category.

        Args:
            category (str): Document category

        Returns:
            list[Context]: context
        """

    @abstractmethod
    def get_context(self, document_name: str, embedding: list[float]) -> list[Context]:
        """Get context from database.

        Args:
            document_name (str): Name of the document
            embedding (list[float]): The embedding

        Returns:
            list[Context]: The context related to the question
        """

    @abstractmethod
    def post_context(
        self,
        text: str,
        document_name: str,
        category: str,
        embedding: list[float],
        document_id: str,
    ) -> bool:
        """Post the curriculum to the database.

        Args:
            text (str): The text to be posted
            document_name (str): Name of the document
            category (str): Category of the document
            embedding (list[float]): The embedding
            document_id (str): Unique identifier for the document

        Returns:
            bool: if the context was posted
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if database is reachable.

        Returns:
            bool: reachable
        """


class MongoDB(Database):
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_COLLECTION]
        self.similarity_threshold = 0.5

    def get_context_by_category(self, category: str) -> list[Context]:
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
                {"$limit": 50},  # optional: limit the number of documents returned
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

    # For backward compatibility
    def get_context_from_npc(self, npc: int) -> list[Context]:
        if not npc:
            raise ValueError("NPC cannot be None")

        # Example using MongoDB Atlas Search with an index named "npc":
        query = {"$search": {"index": "npc", "text": {"path": "npc", "query": npc}}}

        # Execute the aggregate pipeline
        documents = self.collection.aggregate(
            [
                query,
                {"$limit": 50},  # optional: limit the number of documents returned
            ]
        )

        # Convert cursor to a list
        documents = list(documents)
        if not documents:
            raise ValueError(f"No documents found for NPC: {npc}")

        results = []
        for doc in documents:
            results.append(
                Context(
                    text=doc["text"],
                    document_name=doc["document_name"],
                    category=doc.get(
                        "category", f"npc_{doc['npc']}"
                    ),  # Convert old NPC to category format
                )
            )

        return results

    def get_context(self, document_id: str, embedding: list[float]) -> list[Context]:
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

        # Execute the query
        documents = self.collection.aggregate([query])

        if not documents:
            raise ValueError("No documents found")

        # Convert to list
        documents = list(documents)

        results = []

        # Filter out the documents with low similarity
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
                        ),  # Handle both old and new formats
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
        if not text:
            raise ValueError("text cannot be None")

        if not category:
            raise ValueError("Category cannot be None or empty")

        if not document_name:
            raise ValueError("Document name cannot be None")

        if not embedding:
            raise ValueError("Embedding cannot be None")

        try:
            # Insert the curriculum into the database with metadata
            self.collection.insert_one(
                {
                    "text": text,
                    "document_name": document_name,
                    "category": category,  # Using category instead of NPC
                    "embedding": embedding,
                    "document_id": document_id,
                }
            )
            return True
        except Exception as e:
            print("Error in post_context:", e)
            return False

    def is_reachable(self) -> bool:
        try:
            # Send a ping to confirm a successful connection
            self.client.admin.command("ping")
            print("Successfully pinged MongoDB")
            return True
        except Exception as e:
            print(f"Failed to ping MongoDB: {e}")
            return False


class MockDatabase(Database):
    """A mock database for testing purposes, storing data in memory.

    Singleton implementation to ensure only one instance exists.
    """

    _instance = None  # Class variable to hold the singleton instance
    collection = None  # Placeholder for the collection attribute

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # If no instance exists, create one
            cls._instance = super(MockDatabase, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # Initialize only once (avoiding resetting on subsequent calls)
        if not hasattr(self, "initialized"):
            self.data = []  # In-memory storage for mock data
            self.similarity_threshold = 0.7
            self.initialized = True
            self.collection = self  # collection attribute for compatibility

    def get_context_by_category(self, category: str) -> list[Context]:
        if not category:
            raise ValueError("Category cannot be None or empty")

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

    # For backward compatibility
    def get_context_from_npc(self, npc: int) -> list[Context]:
        if not npc:
            raise ValueError("NPC cannot be None")

        results = []
        for document in self.data:
            if document.get("npc") == npc:
                results.append(
                    Context(
                        text=document["text"],
                        document_name=document["document_name"],
                        category=document.get("category", f"npc_{npc}"),
                    )
                )
        return results

    def get_context(self, document_name: str, embedding: list[float]) -> list[Context]:
        if not embedding:
            raise ValueError("Embedding cannot be None")

        results = []

        # Filter documents based on similarity and document_name
        for document in self.data:
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

    def post_context(
        self,
        text: str,
        category: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
    ) -> bool:
        if not text:
            raise ValueError("text cannot be None")
        if not document_id:
            raise ValueError("document_id cannot be None")
        if not category:
            raise ValueError("Category cannot be None or empty")
        if not embedding:
            raise ValueError("embedding cannot be None")

        # Append a new document to the in-memory storage
        self.data.append(
            {
                "text": text,
                "document_name": document_name,
                "category": category,
                "embedding": embedding,
                "document_id": document_id,
            }
        )
        return True

    def is_reachable(self) -> bool:
        return True


class LocalMockDatabase(Database):
    def __init__(self):
        self.data = []
        self.similarity_threshold = 0.7

    def get_context_by_category(self, category: str) -> list[Context]:
        if not category:
            raise ValueError("Category cannot be None or empty")

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

    def get_context(self, document_name: str, embedding: list[float]) -> list[Context]:
        if not embedding:
            raise ValueError("Embedding cannot be None")

        results = []

        # Filter documents based on similarity and document_name
        for document in self.data:
            if document["document_name"] == document_name:
                similarity = similarity_search(embedding, document["embedding"])
                if similarity > self.similarity_threshold:
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
        category: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
    ) -> bool:
        if not text or not document_id or not category or not embedding:
            raise ValueError("All parameters are required and must be valid")

        # Append a new document to the in-memory storage
        self.data.append(
            {
                "text": text,
                "document_name": document_name,
                "category": category,
                "embedding": embedding,
                "document_id": document_id,
            }
        )
        return True

    def is_reachable(self) -> bool:
        return True


def get_database() -> Database:
    """Get the database to use.

    Returns:
        Database: The database to use
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            return MockDatabase()
        case "mongodb":
            return MongoDB()
        case "local_mock":
            return LocalMockDatabase()
        case _:
            raise ValueError("Invalid database type")
