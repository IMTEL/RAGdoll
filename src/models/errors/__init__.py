"""Error domain module exports."""

from src.models.errors.embedding_error import EmbeddingAPIError, EmbeddingError
from src.models.errors.failure import FailureData
from src.models.errors.llm_error import (
    LLMAPIError,
    LLMError,
    LLMGenerationError,
)
from src.models.errors.retrieval_error import RetrievalFailedError


__all__ = [
    "EmbeddingAPIError",
    "EmbeddingError",
    "FailureData",
    "LLMAPIError",
    "LLMError",
    "LLMGenerationError",
    "RetrievalFailedError",
]
