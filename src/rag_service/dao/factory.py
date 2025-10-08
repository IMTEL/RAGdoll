"""Factory functions for creating DAO instances.

This module provides factory functions that instantiate the appropriate
DAO implementation based on configuration settings.
"""

from src.config import Config
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.agent.mongodb_agent_dao import (
    MongoDBAgentDAO,
)
from src.rag_service.dao.context.base import ContextDAO
from src.rag_service.dao.context.mongodb_context_dao import (
    MongoDBContextDAO,
)
from tests.mocks.mock_agent_dao import MockAgentDAO


config = Config()


def get_context_dao() -> ContextDAO:
    """Get the configured context DAO implementation.

    Returns:
        ContextDAO: The DAO instance based on RAG_DATABASE_SYSTEM config

    Raises:
        ValueError: If an invalid database type is configured

    Supported types:
        - 'mongodb': Production MongoDB implementation
        - 'mock': Singleton mock for testing
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            # Import here to avoid circular dependency
            from tests.mocks.mock_context_dao import MockContextDAO

            return MockContextDAO()
        case "mongodb":
            return MongoDBContextDAO()
        case _:
            raise ValueError(
                f"Invalid database type: {config.RAG_DATABASE_SYSTEM}. "
                "Supported types: 'mongodb', 'mock'"
            )


def get_agent_dao() -> AgentDAO:
    """Get the configured agent DAO implementation.

    Returns:
        AgentDAO: The DAO instance based on RAG_DATABASE_SYSTEM config

    Raises:
        ValueError: If an invalid database type is configured

    Supported types:
        - 'mongodb': Production MongoDB implementation
        - 'mock': Mock implementation for testing
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            return MockAgentDAO()
        case "mongodb":
            return MongoDBAgentDAO()
        case _:
            raise ValueError(
                f"Invalid database type: {config.RAG_DATABASE_SYSTEM}. "
                "Supported types: 'mongodb', 'mock'"
            )
