"""Abstract base class for Document DAO pattern."""

from abc import ABC, abstractmethod

from src.models.rag import Document


class DocumentDAO(ABC):
    """Abstract base class for document metadata storage.

    This DAO handles storage and retrieval of document metadata.
    It works in coordination with ContextDAO which stores the actual
    text chunks of the documents.
    """

    @abstractmethod
    def get_by_id(self, document_id: str) -> Document | None:
        """Fetch a document by its unique ID.

        Args:
            document_id (str): Unique identifier for the document

        Returns:
            Document | None: Document if found, None otherwise
        """

    @abstractmethod
    def get_by_agent_id(self, agent_id: str) -> list[Document]:
        """Fetch all documents belonging to a specific agent.

        Args:
            agent_id (str): Agent identifier

        Returns:
            list[Document]: List of documents owned by the agent
        """

    @abstractmethod
    def get_by_agent_and_categories(
        self, agent_id: str, categories: list[str]
    ) -> list[Document]:
        """Fetch documents for an agent filtered by categories.

        Args:
            agent_id (str): Agent identifier
            categories (list[str]): List of category tags to filter by

        Returns:
            list[Document]: Documents matching agent and any of the categories
        """

    @abstractmethod
    def create(self, document: Document) -> Document:
        """Create a new document.

        Args:
            document (Document): Document to create

        Returns:
            Document: Created document with ID populated
        """

    @abstractmethod
    def update(self, document: Document) -> Document:
        """Update an existing document.

        Args:
            document (Document): Document with updated fields

        Returns:
            Document: Updated document

        Raises:
            ValueError: If document does not exist
        """

    @abstractmethod
    def delete(self, document_id: str) -> bool:
        """Delete a document and all its associated contexts.

        This method should cascade delete to remove all context chunks
        associated with this document.

        Args:
            document_id (str): ID of document to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """

    @abstractmethod
    def get_by_name_and_agent(self, name: str, agent_id: str) -> Document | None:
        """Find a document by name within a specific agent.

        Args:
            name (str): Document name
            agent_id (str): Agent identifier

        Returns:
            Document | None: Document if found, None otherwise
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
