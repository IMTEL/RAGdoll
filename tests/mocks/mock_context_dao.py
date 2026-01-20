"""Mock implementation of ContextDAO for testing.

This module provides in-memory mock implementations that mimic the behavior
of a real DAO without requiring actual database connections.
"""

import logging

from src.rag_service.context import Context
from src.rag_service.dao import ContextDAO
from src.utils import singleton


logger = logging.getLogger(__name__)


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

    def _resolve_document_identifiers(
        self, identifiers: list[str] | None, agent_id: str
    ) -> list[str] | None:
        """Resolve document identifiers (names or IDs) to document IDs.
        
        Mock implementation that mimics the MongoDB version's behavior.
        
        Args:
            identifiers: List of document IDs or filenames, or None for all documents
            agent_id: Agent identifier for scoping document lookup
            
        Returns:
            List of resolved document IDs, or None if no filtering requested
        \"\"\"\n        if identifiers is None:\n            return None\n            \n        if len(identifiers) == 0:\n            return []\n        \n        # Common file extensions to detect filenames\n        file_extensions = {\n            '.pdf', '.txt', '.doc', '.docx', '.md', '.csv', '.json', '.xml',\n            '.html', '.htm', '.rtf', '.odt', '.tex', '.log', '.rst'\n        }\n        \n        # Check if any identifier looks like a filename\n        potential_filenames = []\n        confirmed_ids = []\n        \n        for identifier in identifiers:\n            # Check if identifier has a file extension\n            has_extension = any(identifier.lower().endswith(ext) for ext in file_extensions)\n            \n            if has_extension:\n                potential_filenames.append(identifier)\n                logger.debug(f\"Mock: Detected potential filename: {identifier}\")\n            else:\n                confirmed_ids.append(identifier)\n                logger.debug(f\"Mock: Treating as document ID: {identifier}\")\n        \n        # If we found potential filenames, resolve them to IDs\n        if potential_filenames:\n            from src.rag_service.dao.factory import get_document_dao\n            \n            doc_dao = get_document_dao()\n            resolved_docs = doc_dao.get_by_names_and_agent(potential_filenames, agent_id)\n            \n            resolved_ids = [doc.id for doc in resolved_docs]\n            logger.info(\n                f\"Mock: Resolved {len(resolved_ids)} document IDs from {len(potential_filenames)} filenames\"\n            )\n            \n            # Log any filenames that couldn't be resolved\n            resolved_names = {doc.name for doc in resolved_docs}\n            unresolved = set(potential_filenames) - resolved_names\n            if unresolved:\n                logger.warning(\n                    f\"Mock: Could not resolve filenames to documents: {', '.join(unresolved)}\"\n                )\n            \n            # Combine resolved IDs with confirmed IDs\n            return confirmed_ids + resolved_ids\n        \n        # No filenames detected, return as-is\n        return confirmed_ids
"""
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

        # Resolve document names to IDs if filenames are provided
        resolved_document_ids = self._resolve_document_identifiers(
            documents, agent_id
        )

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

            # Filter by documents if provided (now using resolved IDs)
            if resolved_document_ids is not None and document.get("_id") not in resolved_document_ids:
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
