"""Abstract base class for Message Feedback DAO pattern."""

from abc import ABC, abstractmethod

from src.models.feedback import MessageFeedback


class FeedbackDAO(ABC):
    """Abstract base class for message feedback storage.

    The DAO does not check access. Callers must check that the user can
    access the agent, and that ``feedback.agent_id`` matches it, before
    returning or deleting feedback.
    """

    @abstractmethod
    def get_by_id(self, feedback_id: str) -> MessageFeedback | None:
        """Fetch a feedback entry by its unique ID.

        Args:
            feedback_id (str): Unique identifier for the feedback

        Returns:
            MessageFeedback | None: Feedback if found, None otherwise
        """

    @abstractmethod
    def get_recent_by_agent(self, agent_id: str, limit: int) -> list[MessageFeedback]:
        """Fetch the most recent feedback for an agent, newest first.

        Args:
            agent_id (str): Agent identifier
            limit (int): Maximum number of entries to return

        Returns:
            list[MessageFeedback]: Feedback for the agent
        """

    @abstractmethod
    def create(self, feedback: MessageFeedback) -> MessageFeedback:
        """Create a new feedback entry.

        Args:
            feedback (MessageFeedback): Feedback to create

        Returns:
            MessageFeedback: Created feedback with ID and timestamp populated

        Raises:
            ValueError: If required fields are missing
        """

    @abstractmethod
    def delete(self, feedback_id: str) -> bool:
        """Delete a feedback entry.

        Args:
            feedback_id (str): ID of the feedback to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """

    @abstractmethod
    def delete_by_agent(self, agent_id: str) -> int:
        """Delete all feedback for an agent, e.g. when the agent is deleted.

        Args:
            agent_id (str): Agent identifier

        Returns:
            int: Number of deleted entries
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
