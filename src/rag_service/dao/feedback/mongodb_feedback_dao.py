"""MongoDB implementation for message feedback storage."""

import logging
import uuid
from datetime import datetime

from pymongo import ASCENDING, DESCENDING, MongoClient

from src.config import Config
from src.models.feedback import MessageFeedback
from src.rag_service.dao.feedback.base import FeedbackDAO


logger = logging.getLogger(__name__)

config = Config()


class MongoDBFeedbackDAO(FeedbackDAO):
    """MongoDB-backed data access object for message feedback."""

    def __init__(self):
        """Initialize MongoDB connection and create indexes."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_FEEDBACK_COLLECTION]

        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries."""
        try:
            # Every chat turn reads an agent's most recent feedback
            self.collection.create_index(
                [("agent_id", ASCENDING), ("created_at", DESCENDING)]
            )
            logger.info("Feedback collection indexes created successfully")
        except Exception as e:  # noqa: BLE001 - the app works without the index
            logger.warning(f"Could not create indexes: {e}")

    def get_by_id(self, feedback_id: str) -> MessageFeedback | None:
        """Fetch a feedback entry by its unique ID.

        Args:
            feedback_id (str): Unique identifier for the feedback

        Returns:
            MessageFeedback | None: Feedback if found, None otherwise
        """
        if not feedback_id:
            return None

        doc = self.collection.find_one({"_id": feedback_id})
        if not doc:
            return None

        return self._feedback_from_mongo(doc)

    def get_recent_by_agent(self, agent_id: str, limit: int) -> list[MessageFeedback]:
        """Fetch the most recent feedback for an agent, newest first.

        Args:
            agent_id (str): Agent identifier
            limit (int): Maximum number of entries to return

        Returns:
            list[MessageFeedback]: Feedback for the agent
        """
        if not agent_id or limit <= 0:
            return []

        cursor = (
            self.collection.find({"agent_id": agent_id})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )
        return [self._feedback_from_mongo(doc) for doc in cursor]

    def create(self, feedback: MessageFeedback) -> MessageFeedback:
        """Create a new feedback entry.

        Args:
            feedback (MessageFeedback): Feedback to create

        Returns:
            MessageFeedback: Created feedback with ID and timestamp populated

        Raises:
            ValueError: If required fields are missing or the insert fails
        """
        if not feedback.agent_id or not feedback.author_id:
            raise ValueError("Agent ID and author ID are required")

        if not feedback.id:
            feedback.id = str(uuid.uuid4())

        feedback.created_at = datetime.now()

        doc = feedback.model_dump(exclude={"id"})
        doc["_id"] = feedback.id

        try:
            self.collection.insert_one(doc)
            logger.info(
                f"Created feedback '{feedback.id}' for agent '{feedback.agent_id}'"
            )
            return feedback
        except Exception as e:
            logger.error(f"Failed to create feedback: {e}")
            raise ValueError(f"Failed to create feedback: {e}") from e

    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback entry.

        Args:
            feedback_id (str): ID of the feedback to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not feedback_id:
            return False

        try:
            result = self.collection.delete_one({"_id": feedback_id})
        except Exception as e:  # noqa: BLE001 - reported as False
            logger.error(f"Failed to delete feedback: {e}")
            return False

        if result.deleted_count > 0:
            logger.info(f"Deleted feedback '{feedback_id}'")
            return True

        logger.warning(f"Feedback '{feedback_id}' not found for deletion")
        return False

    def delete_by_agent(self, agent_id: str) -> int:
        """Delete all feedback for an agent, e.g. when the agent is deleted.

        Args:
            agent_id (str): Agent identifier

        Returns:
            int: Number of deleted entries
        """
        if not agent_id:
            return 0

        result = self.collection.delete_many({"agent_id": agent_id})
        logger.info(f"Deleted {result.deleted_count} feedback for agent '{agent_id}'")
        return result.deleted_count

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            self.client.admin.command("ping")
            logger.debug("Successfully pinged MongoDB")
            return True
        except Exception as e:  # noqa: BLE001 - reported as unreachable
            logger.error(f"Failed to ping MongoDB: {e}")
            return False

    def _feedback_from_mongo(self, doc: dict) -> MessageFeedback:
        """Convert a MongoDB document to a MessageFeedback model.

        Args:
            doc (dict): MongoDB document

        Returns:
            MessageFeedback: Feedback model instance
        """
        data = {key: value for key, value in doc.items() if key != "_id"}
        return MessageFeedback(id=str(doc["_id"]), **data)
