"""Mock implementation of FeedbackDAO for testing."""

import logging
import uuid
from datetime import datetime

from src.models.feedback import MessageFeedback
from src.rag_service.dao.feedback.base import FeedbackDAO
from src.utils import singleton


logger = logging.getLogger(__name__)


@singleton
class MockFeedbackDAO(FeedbackDAO):
    """In-memory implementation of FeedbackDAO for testing.

    Stores feedback in a dictionary with singleton pattern to
    maintain state across test cases within a session.
    """

    def __init__(self) -> None:
        self._feedback: dict[str, MessageFeedback] = {}

    def get_by_id(self, feedback_id: str) -> MessageFeedback | None:
        """Fetch a feedback entry by its unique ID.

        Args:
            feedback_id (str): Unique identifier for the feedback

        Returns:
            MessageFeedback | None: Feedback if found, None otherwise
        """
        feedback = self._feedback.get(feedback_id)
        return feedback.model_copy(deep=True) if feedback else None

    def get_recent_by_agent(self, agent_id: str, limit: int) -> list[MessageFeedback]:
        """Fetch the most recent feedback for an agent, newest first.

        Args:
            agent_id (str): Agent identifier
            limit (int): Maximum number of entries to return

        Returns:
            list[MessageFeedback]: Feedback for the agent
        """
        if limit <= 0:
            return []
        entries = [f for f in self._feedback.values() if f.agent_id == agent_id]
        entries.sort(key=lambda f: f.created_at, reverse=True)
        return [f.model_copy(deep=True) for f in entries[:limit]]

    def create(self, feedback: MessageFeedback) -> MessageFeedback:
        """Create a new feedback entry.

        Args:
            feedback (MessageFeedback): Feedback to create

        Returns:
            MessageFeedback: Created feedback with ID and timestamp populated

        Raises:
            ValueError: If required fields are missing
        """
        if not feedback.agent_id or not feedback.author_id:
            raise ValueError("Agent ID and author ID are required")

        if not feedback.id:
            feedback.id = str(uuid.uuid4())

        feedback.created_at = datetime.now()

        # Store a copy to avoid external mutations
        self._feedback[feedback.id] = feedback.model_copy(deep=True)

        logger.info(
            f"Mock: Created feedback '{feedback.id}' for agent '{feedback.agent_id}'"
        )
        return feedback

    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback entry.

        Args:
            feedback_id (str): ID of the feedback to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if feedback_id in self._feedback:
            del self._feedback[feedback_id]
            logger.info(f"Mock: Deleted feedback '{feedback_id}'")
            return True
        return False

    def delete_by_agent(self, agent_id: str) -> int:
        """Delete all feedback for an agent, e.g. when the agent is deleted.

        Args:
            agent_id (str): Agent identifier

        Returns:
            int: Number of deleted entries
        """
        ids = [key for key, f in self._feedback.items() if f.agent_id == agent_id]
        for feedback_id in ids:
            del self._feedback[feedback_id]
        return len(ids)

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: Always True for mock implementation
        """
        return True

    def clear(self):
        """Clear all feedback from mock storage.

        Useful for test teardown to ensure clean state.
        """
        self._feedback.clear()
        logger.info("Mock: Cleared all feedback")
