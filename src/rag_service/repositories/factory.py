"""Factory functions for creating repository instances.

This module provides factory functions that instantiate the appropriate
repository implementation based on configuration settings.
"""

from src.config import Config
from src.rag_service.repositories.base import AgentRepository, ContextRepository
from src.rag_service.repositories.mongodb_agent_repository import (
    MongoDBAgentRepository,
)
from src.rag_service.repositories.mongodb_context_repository import (
    MongoDBContextRepository,
)


config = Config()


def get_context_repository() -> ContextRepository:
    """Get the configured context repository implementation.

    Returns:
        ContextRepository: The repository instance based on RAG_DATABASE_SYSTEM config

    Raises:
        ValueError: If an invalid database type is configured

    Supported types:
        - 'mongodb': Production MongoDB implementation
        - 'mock': Singleton mock for testing
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            # Import here to avoid circular dependency
            from tests.mocks.mock_context_repository import MockContextRepository

            return MockContextRepository()
        case "mongodb":
            return MongoDBContextRepository()
        case _:
            raise ValueError(
                f"Invalid database type: {config.RAG_DATABASE_SYSTEM}. "
                "Supported types: 'mongodb', 'mock'"
            )


def get_agent_repository() -> AgentRepository:
    """Get the configured agent repository implementation.

    Returns:
        AgentRepository: The repository instance based on RAG_DATABASE_SYSTEM config

    Raises:
        ValueError: If an invalid database type is configured

    Supported types:
        - 'mongodb': Production MongoDB implementation
        - 'mock': Mock implementation for testing
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            # Import here to avoid circular dependency
            from tests.mocks.mock_agent_repository import MockAgentRepository

            return MockAgentRepository()
        case "mongodb":
            return MongoDBAgentRepository()
        case _:
            raise ValueError(
                f"Invalid database type: {config.RAG_DATABASE_SYSTEM}. "
                "Supported types: 'mongodb', 'mock'"
            )
