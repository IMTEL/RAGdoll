"""Mock implementation of DocumentDAO for testing."""

import logging
from datetime import datetime

from src.models.rag import Document
from src.rag_service.dao.document.base import DocumentDAO
from src.utils import singleton


logger = logging.getLogger(__name__)


@singleton
class MockDocumentDAO(DocumentDAO):
    """In-memory implementation of DocumentDAO for testing.

    Stores documents in a dictionary with singleton pattern to
    maintain state across test cases within a session.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def get_by_id(self, document_id: str) -> Document | None:
        """Fetch a document by its unique ID.

        Args:
            document_id (str): Unique identifier for the document

        Returns:
            Document | None: Document if found, None otherwise
        """
        return self._documents.get(document_id)

    def get_by_agent_id(self, agent_id: str) -> list[Document]:
        """Fetch all documents belonging to a specific agent.

        Args:
            agent_id (str): Agent identifier

        Returns:
            list[Document]: List of documents owned by the agent
        """
        return [doc for doc in self._documents.values() if doc.agent_id == agent_id]

    def create(self, document: Document) -> Document:
        """Create a new document.

        Args:
            document (Document): Document to create

        Returns:
            Document: Created document with ID populated

        Raises:
            ValueError: If document already exists or required fields missing
        """
        if not document.name:
            raise ValueError("Document name is required")
        if not document.agent_id:
            raise ValueError("Agent ID is required")

        # Check if document with same name exists for this agent
        existing = self.get_by_name_and_agent(document.name, document.agent_id)
        if existing:
            raise ValueError(
                f"Document '{document.name}' already exists for agent '{document.agent_id}'"
            )

        # Generate ID if not provided
        if not document.id:
            import uuid

            document.id = str(uuid.uuid4())

        now = datetime.now()
        document.created_at = now
        document.updated_at = now

        # Store a copy to avoid external mutations
        self._documents[document.id] = document.model_copy(deep=True)

        logger.info(
            f"Mock: Created document '{document.name}' with ID '{document.id}' for agent '{document.agent_id}'"
        )
        return document

    def update(self, document: Document) -> Document:
        """Update an existing document.

        Args:
            document (Document): Document with updated fields

        Returns:
            Document: Updated document

        Raises:
            ValueError: If document does not exist
        """
        if not document.id:
            raise ValueError("Document ID is required for update")

        if document.id not in self._documents:
            raise ValueError(f"Document with ID '{document.id}' not found")

        # Update timestamp on a copy to avoid mutating the input
        updated_doc = document.model_copy(deep=True)
        updated_doc.updated_at = datetime.now()

        # Store the copy
        self._documents[document.id] = updated_doc

        logger.info(f"Mock: Updated document '{document.id}'")
        return updated_doc

    def delete(self, document_id: str) -> bool:
        """Delete a document.

        Note: In mock, this doesn't cascade delete contexts since
        MockContextDAO is separate. Tests should handle context cleanup.

        Args:
            document_id (str): ID of document to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if document_id in self._documents:
            del self._documents[document_id]
            logger.info(f"Mock: Deleted document '{document_id}'")
            return True
        return False

    def get_by_name_and_agent(self, name: str, agent_id: str) -> Document | None:
        """Find a document by name within a specific agent.

        Args:
            name (str): Document name
            agent_id (str): Agent identifier

        Returns:
            Document | None: Document if found, None otherwise
        """
        for doc in self._documents.values():
            if doc.name == name and doc.agent_id == agent_id:
                return doc
        return None

    def get_by_names_and_agent(self, names: list[str], agent_id: str) -> list[Document]:
        """Find multiple documents by names within a specific agent.

        Args:
            names (list[str]): List of document names to find
            agent_id (str): Agent identifier

        Returns:
            list[Document]: List of matching documents (may be fewer than requested names)
        """
        if not names or not agent_id:
            return []
        
        return [
            doc for doc in self._documents.values()
            if doc.name in names and doc.agent_id == agent_id
        ]

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: Always True for mock implementation
        """
        return True

    def clear(self):
        """Clear all documents from mock storage.

        Useful for test teardown to ensure clean state.
        """
        self._documents.clear()
        logger.info("Mock: Cleared all documents")
