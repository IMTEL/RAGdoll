"""Abstract base classes for repository pattern.

This module defines the interfaces for document context and agent repositories.
"""

from abc import ABC, abstractmethod

from src.models.agent import Agent


class AgentRepository(ABC):
    """Abstract base class for agent storage.

    This repository handles persistence of AI agent configurations.
    """

    @abstractmethod
    def create_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration.

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
    def is_reachable(self) -> bool:
        """Check if the repository backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """

    @abstractmethod
    def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent configuration.

        Args:
            agent (Agent): The agent object with updated fields

        Returns:
            Agent: The updated agent object
        """
