"""DAO pattern implementations for data access.

This package provides abstraction layers for database operations following
the DAO pattern, separating business logic from data access concerns.
"""

from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.agent.mongodb_agent_dao import (
    MongoDBAgentDAO,
)
from src.rag_service.dao.context.base import ContextDAO
from src.rag_service.dao.context.mongodb_context_dao import (
    MongoDBContextDAO,
)
from src.rag_service.dao.factory import (
    get_agent_dao,
    get_context_dao,
)


# Backward compatibility - old names
get_database = get_context_dao

__all__ = [
    "AgentDAO",
    # Base classes
    "ContextDAO",
    "MongoDBAgentDAO",
    # MongoDB implementations
    "MongoDBContextDAO",
    "get_agent_dao",
    # Factory functions
    "get_context_dao",
]
