"""Error domain module exports."""

from src.models.errors.failure import FailureData
from src.models.errors.llm_error import (
    LLMAPIError,
    LLMError,
    LLMGenerationError,
)


__all__ = [
    "FailureData",
    "LLMAPIError",
    "LLMError",
    "LLMGenerationError",
]
