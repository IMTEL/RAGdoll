"""Repository pattern implementations for data access.

This package provides abstraction layers for database operations following
the Repository pattern, separating business logic from data access concerns.
"""

from src.rag_service.repositories.base import AgentRepository, ContextRepository
from src.rag_service.repositories.factory import (
    get_agent_repository,
    get_context_repository,
)
from src.rag_service.repositories.mongodb_agent_repository import (
    MongoDBAgentRepository,
)
from src.rag_service.repositories.mongodb_context_repository import (
    MongoDBContextRepository,
)


# Backward compatibility - old names
get_database = get_context_repository
get_agent_database = get_agent_repository

__all__ = [
    "AgentRepository",
    # Base classes
    "ContextRepository",
    "MongoDBAgentRepository",
    # MongoDB implementations
    "MongoDBContextRepository",
    "get_agent_database",  # backward compatibility
    "get_agent_repository",
    # Factory functions
    "get_context_repository",
    "get_database",  # backward compatibility
]
