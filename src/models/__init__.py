"""Domain models for RAGdoll application.

This package contains business domain models organized by functional area:
- agents: AI agent configurations and roles
- chat: Message and conversation models
- training: Progress tracking for educational tasks
- rag: Retrieval-Augmented Generation models
- errors: Error and failure tracking
"""

from src.models.agents import Agent, Role
from src.models.chat import Message
from src.models.errors import FailureData
from src.models.rag import RAGPostModel
from src.models.training import (
    ListProgressData,
    ProgressData,
    StepProgressDTO,
    SubtaskProgressDTO,
)


__all__ = [
    "Agent",
    "FailureData",
    "ListProgressData",
    "Message",
    "ProgressData",
    "RAGPostModel",
    "Role",
    "StepProgressDTO",
    "SubtaskProgressDTO",
]
