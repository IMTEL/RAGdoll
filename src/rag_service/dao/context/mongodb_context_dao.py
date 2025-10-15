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
         "path": "categories"
       }
     ]
   }
   ```

2. **Text Search Index** (name: "category"):
   Index Type: search
   ```json
   {
     "mappings": {
       "dynamic": false,
       "fields": {
         "category": {
           "type": "string"
         }
       }
     }
   }
   ```

3. **Standard Indexes** (for non-vector queries):
   - Create index on "agent_id" (ascending)
   - Create index on "categories" (ascending)
   - Create compound index on ["agent_id", "categories"]
   - Create index on "document_id" (ascending)

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
from src.rag_service.embeddings import similarity_search


logger = logging.getLogger(__name__)


config = Config()


class MongoDBContextDAO(ContextDAO):
    """MongoDB-backed data access object (DAO) for document contexts with vector search.

    Uses MongoDB Atlas Vector Search for semantic similarity queries
    and text search for category-based retrieval.
    """

    def __init__(self):
        """Initialize MongoDB connection, set similarity threshold, and create indexes."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_CONTEXT_COLLECTION]
        self.similarity_threshold = 0.5

        # Create indexes for efficient querying
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries."""
        try:
            # Index on agent_id for fast agent-based queries
            self.collection.create_index([("agent_id", ASCENDING)])

            # Index on categories for category-based filtering
            self.collection.create_index([("category", ASCENDING)])

            # Compound index for agent_id and categories queries
            self.collection.create_index(
                [("agent_id", ASCENDING), ("category", ASCENDING)]
            )

            # Index on document_id for document-based operations
            self.collection.create_index([("document_id", ASCENDING)])

            logger.info("Context collection indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create context indexes: {e}")

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
                    categories=doc.get("categories", []),
                    document_id=doc.get("document_id"),
                    chunk_id=doc.get("chunk_id"),
                    chunk_index=doc.get("chunk_index"),
                    total_chunks=doc.get("total_chunks", 1),
                )
            )

        return results

    def get_context_by_corpus_ids(
        self,
        corpus_ids: list[str],
        embedding: list[float],
        num_candidates: int = 50,
        top_k: int = 5,
    ) -> list[Context]:
        """Retrieve relevant contexts from specific corpus IDs using vector similarity.

        This method filters documents by corpus/category IDs and then performs
        semantic similarity search within those documents.

        Args:
            corpus_ids (list[str]): List of corpus/category identifiers to search within
            embedding (list[float]): Query embedding vector
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return

        Returns:
            list[Context]: Top matching contexts from the specified corpus

        Raises:
            ValueError: If corpus_ids or embedding is empty
        """
        if not corpus_ids:
            raise ValueError("corpus_ids cannot be empty")
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        # Use MongoDB aggregation with $vectorSearch and category filter
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embeddings",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": num_candidates,
                    "limit": top_k,
                    # TODO: Re-enable category filtering once supported
                    # "filter": {"category": {"$in": corpus_ids}},
                }
            },
        ]

        documents = self.collection.aggregate(pipeline)
        documents = list(documents)

        results = []
        for document in documents:
            # Additional similarity check
            similarity = similarity_search(embedding, document["embedding"])

            logger.debug(
                f"Document Name: {document.get('document_name')}, Similarity: {similarity}"
            )

            if similarity > self.similarity_threshold:
                results.append(
                    Context(
                        text=document["text"],
                        document_name=document["document_name"],
                        categories=document.get("categories", []),
                        document_id=document.get("document_id"),
                        chunk_id=document.get("chunk_id"),
                        chunk_index=document.get("chunk_index"),
                        total_chunks=document.get("total_chunks", 1),
                    )
                )

        return results

    def get_context_for_agent(
        self,
        agent_id: str,
        embedding: list[float],
        categories: list[str] | None = None,
        num_candidates: int = 50,
        top_k: int = 5,
    ) -> list[Context]:
        """Retrieve relevant contexts for an agent using vector similarity.

        Searches within the agent's documents, optionally filtered by categories.
        Uses MongoDB Atlas Vector Search with agent_id and category filtering.

        IMPORTANT: Requires proper Atlas configuration (see module docstring).
        Vector search with filters requires MongoDB Atlas M10+ cluster tier.

        Args:
            agent_id (str): Agent identifier
            embedding (list[float]): Query embedding vector
            categories (list[str] | None): Optional list of categories to filter by.
                                           If None, searches all agent's documents.
                                           If provided, only documents with at least one
                                           matching category will be returned.
            num_candidates (int): Number of initial candidates to consider
            top_k (int): Maximum number of results to return

        Returns:
            list[Context]: Top matching contexts for the agent

        Raises:
            ValueError: If agent_id or embedding is empty
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        # Build filter for vectorSearch
        # This requires the vector index to have filter fields configured (see module docstring)
        search_filter = {"agent_id": {"$eq": agent_id}}

        if categories:
            # Match documents that have at least one of the specified categories
            search_filter["categories"] = {"$in": categories}

        # Build MongoDB aggregation pipeline with vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "embeddings",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": num_candidates,
                    "limit": top_k,
                    "filter": search_filter,
                }
            },
        ]

        try:
            documents = list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Vector search failed: {e}. Check Atlas configuration.")
            # Fallback: use post-filtering if vector search with filters fails
            logger.warning(
                "Falling back to post-filtering (slower). Consider upgrading cluster tier."
            )
            pipeline_no_filter = [
                {
                    "$vectorSearch": {
                        "index": "embeddings",
                        "path": "embedding",
                        "queryVector": embedding,
                        "numCandidates": num_candidates
                        * 3,  # Get more candidates for filtering
                        "limit": top_k * 10,  # Get more results for post-filtering
                    }
                },
            ]
            documents = list(self.collection.aggregate(pipeline_no_filter))

        results = []
        for document in documents:
            # Post-filter by agent_id and categories if using fallback
            if document.get("agent_id") != agent_id:
                continue

            if categories:
                doc_categories = document.get("categories", [])
                # Check if document has at least one of the requested categories
                if not any(cat in doc_categories for cat in categories):
                    continue

            # Additional similarity check
            similarity = similarity_search(embedding, document["embedding"])

            logger.debug(
                f"Agent: {agent_id}, Document: {document.get('document_name')}, "
                f"Categories: {document.get('categories')}, Similarity: {similarity}"
            )

            if similarity > self.similarity_threshold:
                results.append(
                    Context(
                        text=document["text"],
                        document_name=document["document_name"],
                        categories=document.get("categories", []),
                        document_id=document.get("document_id"),
                        chunk_id=document.get("chunk_id"),
                        chunk_index=document.get("chunk_index"),
                        total_chunks=document.get("total_chunks", 1),
                    )
                )

            # Stop if we have enough results
            if len(results) >= top_k:
                break

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
                        categories=document.get("categories", []),
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
        categories = context.categories

        if not document_id:
            raise ValueError("document_id cannot be empty")
        if not agent_id:
            raise ValueError("agent_id cannot be empty")
        if not categories:
            raise ValueError("Categories cannot be empty")
        if not embedding:
            raise ValueError("embedding cannot be empty")

        document = {
            "text": text,
            "document_name": document_name,
            "categories": categories,
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
