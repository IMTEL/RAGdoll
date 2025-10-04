"""Domain models for RAGdoll application.

This package contains business domain models organized by functional area:
- agents: AI agent configurations and roles
- chat: Message and conversation models
- training: Progress tracking for educational tasks
- rag: Retrieval-Augmented Generation models
- errors: Error and failure tracking
"""

from src.domain.agents import Agent, Role
from src.domain.chat import Message
from src.domain.errors import FailureData
from src.domain.rag import RAGPostModel
from src.domain.training import (
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
