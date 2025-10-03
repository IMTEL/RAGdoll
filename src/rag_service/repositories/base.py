"""Abstract base classes for repository pattern.

This module defines the interfaces for document context and agent repositories.
"""

from abc import ABC, abstractmethod

from src.models.agent import Agent
from src.rag_service.context import Context


class ContextRepository(ABC):
    """Abstract base class for document context storage.

    This repository handles storage and retrieval of document contexts
    with embeddings for semantic search.
    """

    @classmethod
    def __instancecheck__(cls, instance: any) -> bool:
        return cls.__subclasscheck__(type(instance))

    @classmethod
    def __subclasscheck__(cls, subclass: any) -> bool:
        return (
            hasattr(subclass, "get_context") and callable(subclass.get_context)
        ) and (hasattr(subclass, "post_context") and callable(subclass.post_context))

    @abstractmethod
    def get_context_by_category(self, category: str) -> list[Context]:
        """Fetch all contexts associated with the given category.

        Args:
            category (str): Document category to filter by

        Returns:
            list[Context]: List of contexts matching the category
        """

    @abstractmethod
    def get_context(self, document_id: str, embedding: list[float]) -> list[Context]:
        """Retrieve context using semantic similarity search.

        Args:
            document_id (str): Identifier for the document
            embedding (list[float]): Query embedding vector for similarity search

        Returns:
            list[Context]: Most relevant contexts based on embedding similarity
        """

    @abstractmethod
    def post_context(
        self,
        text: str,
        document_name: str,
        category: str,
        embedding: list[float],
        document_id: str,
    ) -> bool:
        """Store a new context document with its embedding.

        Args:
            text (str): The text content to store
            document_name (str): Name of the source document
            category (str): Category/classification of the document
            embedding (list[float]): Vector embedding of the text
            document_id (str): Unique identifier for the document

        Returns:
            bool: True if successfully stored, False otherwise
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the repository backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """


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
