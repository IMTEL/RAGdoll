"""MongoDB implementation for context document storage.

LOCAL VECTOR SEARCH WITH FAISS:
================================

This system uses FAISS (Facebook AI Similarity Search) for local vector search
instead of MongoDB Atlas Vector Search. This makes the system self-contained and
works with MongoDB Community Edition.

1. **Vector Search**: Uses FAISS with cosine similarity
   - Embeddings are indexed in FAISS locally per agent
   - Supports fast cosine similarity search without external APIs
   - Vectors are persisted locally: /data/faiss_indices/agent_<agent_id>.index

2. **Keyword Search**: Uses MongoDB full-text search (BM25)
   Index Type: search
   Manual index creation:
   ```json
   {
     "mappings": {
       "dynamic": false,
       "fields": {
         "text": {"type": "string"},
         "agent_id": {"type": "string"},
         "document_id": {"type": "string"}
       }
     }
   }
   ```

3. **Standard Indexes** (created automatically):
   - Index on "agent_id" (ascending)
   - Index on "document_id" (ascending)
   - Compound index on ["agent_id", "document_id"]

Notes:
- Works with MongoDB Community Edition
- Vector search is fully self-contained in FAISS
- No Atlas M10+ cluster required
"""

import logging

from pymongo import ASCENDING, MongoClient

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.dao.context.base import ContextDAO
from src.rag_service.dao.context.hybrid_search import hybrid_search
from src.rag_service.dao.context.faiss_vector_store import FAISSVectorStoreManager
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
        self.similarity_threshold = 0.5  # lower = more similar

        # Create indexes for efficient querying
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries and vector search."""
        try:
            # Index on agent_id for fast agent-based queries
            self.collection.create_index([("agent_id", ASCENDING)])

            # Index on document_id for document-based operations
            self.collection.create_index([("document_id", ASCENDING)])

            # Compound index for agent_id and document_id queries
            self.collection.create_index(
                [("agent_id", ASCENDING), ("document_id", ASCENDING)]
            )

            # logger.info("Context collection indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create context indexes: {e}")

        # Note: Vector search is handled locally with FAISS
        # Keyword search is handled locally with BM25
        # No Atlas Search indexes needed for self-contained deployment

    def _create_vector_search_index(self):
        """Create Atlas Vector Search index (disabled for local deployment).

        For self-contained deployments using local FAISS, this is not needed.
        Vector search is handled by FAISSVectorStoreManager instead.
        """
        logger.debug(
            "Skipping Atlas vector search index creation. "
            "Using local FAISS for vector search instead."
        )

    def _create_keyword_search_index(self):
        """Create Atlas Search index (disabled for local deployment).

        For self-contained deployments using local BM25, this is not needed.
        Keyword search is handled by the BM25 implementation in hybrid_search.py instead.
        """
        logger.debug(
            "Skipping Atlas keyword search index creation. "
            "Using local BM25 for keyword search instead."
        )

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
            similarity_threshold (float | None): Minimum similarity score for results.
                                                  If None, uses instance default.
            hybrid_search_alpha (float | None): Weight for hybrid search (0=keyword, 1=vector).
                                                 If None, uses default value of 0.75.

        Returns:
            list[Context]: Top matching contexts for the agent

        Raises:
            ValueError: If agent_id or embedding is empty
        """
        available_documents = documents if documents is not None else []
        # If an explicit empty list of documents is provided, there are no
        # accessible documents for this agent. Returning early avoids
        # constructing MongoDB queries like {"document_id": {"$in": []}}
        # which cause an OperationFailure in some MongoDB versions.
        if documents is not None and len(documents) == 0:
            logger.warning(
                f"No accessible documents for agent {agent_id}. "
                "The active role has no documents assigned to it. "
                "Please assign documents to this role in the agent configuration."
            )
            return []
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not query_embedding:
            raise ValueError("Embedding cannot be empty")
        print("Available documents for filtering:", available_documents)
        logger.info(f"Searching in {len(available_documents)} documents for agent {agent_id}")

        keyword_text = (
            keyword_query_text if keyword_query_text is not None else query_text
        )
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.similarity_threshold
        )
        alpha = hybrid_search_alpha if hybrid_search_alpha is not None else 0.75
        adjusted_num_candidates = max(num_candidates, top_k * 3)

        results = hybrid_search(
            alpha,
            agent_id,
            query_embedding,
            query_text,
            keyword_text,
            available_documents,
            self.collection,
            threshold,
            num_candidates=adjusted_num_candidates,
            top_k=top_k,
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
        """Store a new context document with metadata and embedding.
        
        Stores the document in MongoDB and also indexes the embedding in FAISS
        for local vector similarity search.
        """
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
        
        # Also add to FAISS vector store for local similarity search
        try:
            manager = FAISSVectorStoreManager()
            chunk_id = context.chunk_id or f"{document_id}_{context.chunk_index}"
            manager.add_vector(
                agent_id=agent_id,
                chunk_id=chunk_id,
                embedding=embedding,
                document_id=document_id,
                document_name=document_name,
                text=text,
                chunk_index=context.chunk_index,
                total_chunks=context.total_chunks,
                embedding_dim=len(embedding),
            )
            logger.debug(f"Added embedding to FAISS for chunk {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to add embedding to FAISS: {e}")
            # Don't fail the entire operation if FAISS write fails
        
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
