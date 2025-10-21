"""MongoDB implementation for context document storage.

MONGODB ATLAS VECTOR SEARCH CONFIGURATION GUIDE:
================================================

To enable vector search and filtering capabilities, configure the following indexes in MongoDB Atlas:

1. **Vector Search Index** (name: "embeddings"):
   Index Type: vectorSearch
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 768,  // Adjust based on your embedding model (e.g., 768 for Google, 1536 for OpenAI)
         "similarity": "cosine"
       },
       {
         "type": "filter",
         "path": "agent_id"
       },
       {
         "type": "filter",
         "path": "document_id"
       }
     ]
   }
   ```

2. **Standard Indexes** (for non-vector queries):
   - Create index on "agent_id" (ascending)
   - Create index on "document_id" (ascending)
   - Create compound index on ["agent_id", "document_id"]

How to create indexes in MongoDB Atlas:
1. Go to your Atlas cluster
2. Click "Browse Collections"
3. Select your database and collection
4. Click "Search Indexes" tab
5. Click "Create Search Index" or "Create Index"
6. Choose "JSON Editor" and paste the appropriate configuration above

Note: Vector search with filters requires MongoDB Atlas M10+ cluster tier.
"""

import logging

from pymongo import ASCENDING, MongoClient

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.dao.context.base import ContextDAO
from src.rag_service.dao.context.hybrid_search import hybrid_search
from src.rag_service.embeddings import similarity_search


logger = logging.getLogger(__name__)


config = Config()


class MongoDBContextDAO(ContextDAO):
    """MongoDB-backed data access object (DAO) for document contexts with vector search.

    Uses MongoDB Atlas Vector Search for semantic similarity queries.
    """

    def __init__(self):
        """Initialize MongoDB connection, set similarity threshold, and create indexes."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_CONTEXT_COLLECTION]
        self.similarity_threshold = 0.5 # lower = more similar

        # Create indexes for efficient querying
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries."""
        try:
            # Index on agent_id for fast agent-based queries
            self.collection.create_index([("agent_id", ASCENDING)])

            # Index on document_id for document-based operations
            self.collection.create_index([("document_id", ASCENDING)])

            # Compound index for agent_id and document_id queries
            self.collection.create_index(
                [("agent_id", ASCENDING), ("document_id", ASCENDING)]
            )

            logger.info("Context collection indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create context indexes: {e}")

    def get_context_for_agent(
        self,
        agent_id: str,
        query_embedding: list[float],
        query_text: str,
        keyword_query_text: str | None = None,
        documents: list[str] | None = None,
        num_candidates: int = 50,
        top_k: int = 5,
    ) -> list[Context]:
        """Retrieve relevant contexts for an agent using vector similarity.

        Searches within the agent's documents, filtered by document IDs.
        Uses MongoDB Atlas Vector Search with agent_id and document_id filtering.

        IMPORTANT: Requires proper Atlas configuration (see module docstring).
        Vector search with filters requires MongoDB Atlas M10+ cluster tier.

        Args:
            agent_id (str): Agent identifier
            query_embedding (list[float]): Query embedding vector
            query_text (str): Full query text for vector search (includes context)
            keyword_query_text (str | None): Simplified query text for BM25 keyword search.
                                              If None, uses query_text for both searches.
            documents (list[str] | None): Optional list of document IDs to filter by.
                                           If None, searches all agent's documents.
                                           If provided, only documents with at least one
                                           matching ID will be returned.
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return

        Returns:
            list[Context]: Top matching contexts for the agent

        Raises:
            ValueError: If agent_id or embedding is empty
        """
        available_documents = documents if documents is not None else []
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not query_embedding:
            raise ValueError("Embedding cannot be empty")
        print("Available documents for filtering:", available_documents)
        
        # Use keyword_query_text if provided, otherwise fall back to query_text
        keyword_text = keyword_query_text if keyword_query_text is not None else query_text
        
        results = hybrid_search(
                    0.9, # 90% vector, 10% keyword
                    agent_id,
                    query_embedding, 
                    query_text,
                    keyword_text,
                    available_documents,
                    self.collection,
                    self.similarity_threshold,
                    num_candidates=num_candidates,
                    top_k=top_k)

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
        """Store a new context document with metadata and embedding."""
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

        self.collection.insert_one(document)
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
