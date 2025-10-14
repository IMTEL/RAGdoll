"""MongoDB implementation for document metadata storage."""

import logging
from datetime import datetime

from pymongo import ASCENDING, MongoClient

from src.config import Config
from src.models.rag import Document
from src.rag_service.dao.context.mongodb_context_dao import MongoDBContextDAO
from src.rag_service.dao.document.base import DocumentDAO


logger = logging.getLogger(__name__)

config = Config()


class MongoDBDocumentDAO(DocumentDAO):
    """MongoDB-backed data access object for document metadata.

    Manages document metadata and coordinates with ContextDAO for
    cascading operations. Creates indexes for efficient querying.
    """

    def __init__(self):
        """Initialize MongoDB connection and create indexes."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_DOCUMENTS_COLLECTION]
        self.context_dao = MongoDBContextDAO()

        # Create indexes for efficient querying
        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries."""
        try:
            # Index on agent_id for fast agent-based queries
            self.collection.create_index([("agent_id", ASCENDING)])

            # Index on categories for category-based filtering
            self.collection.create_index([("categories", ASCENDING)])

            # Compound index for agent_id and categories queries
            self.collection.create_index(
                [("agent_id", ASCENDING), ("categories", ASCENDING)]
            )

            # Unique index on name within an agent
            self.collection.create_index(
                [("agent_id", ASCENDING), ("name", ASCENDING)], unique=True
            )

            logger.info("Document collection indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def get_by_id(self, document_id: str) -> Document | None:
        """Fetch a document by its unique ID.

        Args:
            document_id (str): Unique identifier for the document

        Returns:
            Document | None: Document if found, None otherwise
        """
        if not document_id:
            return None

        doc = self.collection.find_one({"_id": document_id})
        if not doc:
            return None

        return self._doc_from_mongo(doc)

    def get_by_agent_id(self, agent_id: str) -> list[Document]:
        """Fetch all documents belonging to a specific agent.

        Args:
            agent_id (str): Agent identifier

        Returns:
            list[Document]: List of documents owned by the agent
        """
        if not agent_id:
            return []

        cursor = self.collection.find({"agent_id": agent_id})
        return [self._doc_from_mongo(doc) for doc in cursor]

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
        if not agent_id:
            return []

        query = {"agent_id": agent_id}
        if categories:
            # Match documents that have any of the specified categories
            query["categories"] = {"$in": categories}  # type: ignore[assignment]

        cursor = self.collection.find(query)
        return [self._doc_from_mongo(doc) for doc in cursor]

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

        doc_dict = {
            "_id": document.id,
            "name": document.name,
            "agent_id": document.agent_id,
            "categories": document.categories,
            "created_at": document.created_at,
            "updated_at": document.updated_at,
        }

        try:
            self.collection.insert_one(doc_dict)
            logger.info(
                f"Created document '{document.name}' with ID '{document.id}' for agent '{document.agent_id}'"
            )
            return document
        except Exception as e:
            logger.error(f"Failed to create document: {e}")
            raise ValueError(f"Failed to create document: {e}") from e

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

        existing = self.get_by_id(document.id)
        if not existing:
            raise ValueError(f"Document with ID '{document.id}' not found")

        document.updated_at = datetime.now()

        update_dict = {
            "$set": {
                "name": document.name,
                "categories": document.categories,
                "updated_at": document.updated_at,
            }
        }

        try:
            result = self.collection.update_one({"_id": document.id}, update_dict)
            if result.modified_count == 0:
                logger.warning(f"No changes made to document '{document.id}'")
            else:
                logger.info(f"Updated document '{document.id}'")
            return document
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise ValueError(f"Failed to update document: {e}") from e

    def delete(self, document_id: str) -> bool:
        """Delete a document and all its associated contexts.

        This method cascades delete to remove all context chunks
        associated with this document.

        Args:
            document_id (str): ID of document to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not document_id:
            return False

        try:
            # First, delete all contexts associated with this document
            context_collection = self.context_dao.collection
            context_result = context_collection.delete_many(
                {"document_id": document_id}
            )
            logger.info(
                f"Deleted {context_result.deleted_count} contexts for document '{document_id}'"
            )

            # Then delete the document itself
            result = self.collection.delete_one({"_id": document_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted document '{document_id}'")
                return True
            else:
                logger.warning(f"Document '{document_id}' not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def get_by_name_and_agent(self, name: str, agent_id: str) -> Document | None:
        """Find a document by name within a specific agent.

        Args:
            name (str): Document name
            agent_id (str): Agent identifier

        Returns:
            Document | None: Document if found, None otherwise
        """
        if not name or not agent_id:
            return None

        doc = self.collection.find_one({"name": name, "agent_id": agent_id})
        if not doc:
            return None

        return self._doc_from_mongo(doc)

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            self.client.admin.command("ping")
            logger.debug("Successfully pinged MongoDB")
            return True
        except Exception as e:
            logger.error(f"Failed to ping MongoDB: {e}")
            return False

    def _doc_from_mongo(self, doc: dict) -> Document:
        """Convert MongoDB document to Document model.

        Args:
            doc (dict): MongoDB document

        Returns:
            Document: Document model instance
        """
        return Document(
            id=doc["_id"],
            name=doc["name"],
            agent_id=doc["agent_id"],
            categories=doc.get("categories", []),
            created_at=doc.get("created_at", datetime.now()),
            updated_at=doc.get("updated_at", datetime.now()),
        )
