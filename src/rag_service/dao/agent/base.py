"""Abstract base classes for DAO pattern.

This module defines the interfaces for document context and agent DAOs.
"""

from abc import ABC, abstractmethod
from datetime import datetime

from src.models.accesskey import AccessKey
from src.models.agent import Agent


class AgentDAO(ABC):
    """Abstract base class for agent storage.

    This DAO handles persistence of AI agent configurations.
    """

    @abstractmethod
    def add_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration and updates if the agent already exists.

        On update the stored ``access_key`` list is left as it is: keys are
        only changed through push_access_key, pull_access_keys and
        touch_access_key, so an update from an older read cannot drop a key
        that was minted in the meantime.

        Args:
            agent (Agent): The agent object to persist

        Returns:
            Agent: The stored agent object
        """

    @abstractmethod
    def get_agents(self) -> list[Agent]:
        """Retrieve all stored agent configurations.

        Returns:
            list[Agent]: List of all agent configurations
        """

    @abstractmethod
    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """Retrieve a specific agent by ID.

        Args:
            agent_id (str): The unique identifier of the agent

        Returns:
            Agent | None: The agent if found, None otherwise
        """

    @abstractmethod
    def touch_access_key(self, agent_id: str, key_id: str, when: datetime) -> None:
        """Record ``when`` as the last use of one of the agent's access keys."""

    @abstractmethod
    def push_access_key(self, agent_id: str, access_key: AccessKey) -> bool:
        """Atomically append one access key to the agent.

        Returns:
            bool: False if the agent does not exist
        """

    @abstractmethod
    def pull_access_keys(self, agent_id: str, key_ids: list[str]) -> bool:
        """Atomically remove the agent's access keys with these ids.

        Returns:
            bool: True if at least one key was removed
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """

    @abstractmethod
    def delete_agent_by_id(self, agent_id: str) -> bool:
        """Delete a specific agent by ID.

        Args:
            agent_id (str): The MongoDB ObjectId as a string

        Returns:
            bool: True if the agent was deleted successfully, False otherwise
        """
