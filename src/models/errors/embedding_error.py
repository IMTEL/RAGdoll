"""Custom exceptions for embedding-related errors."""


class EmbeddingAPIError(Exception):
    """Raised when there's an API authentication or access issue with the embedding provider.

    ``status_code`` is the HTTP status a chat request reports for it: the
    provider's own status (401/402/403/429) when the original error carries
    one, 429 for quota or rate-limit messages, otherwise 401.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        original_error: Exception | str,
        status_code: int | None = None,
    ):
        self.provider = provider
        self.model = model
        self.original_error = original_error

        # Create a more user-friendly message
        error_str = str(original_error)
        error_lower = error_str.lower()
        is_quota = (
            "quota exceeded" in error_lower
            or "rate limit" in error_lower
            or "429" in error_lower
        )
        provider_status = getattr(original_error, "status_code", None)
        if status_code is not None:
            self.status_code = status_code
        elif provider_status in (401, 402, 403, 429):
            self.status_code = provider_status
        elif is_quota:
            self.status_code = 429
        else:
            self.status_code = 401

        if is_quota:
            # Quota/Rate limit error
            message = (
                f"{provider} API quota exceeded for model '{model}'. "
                f"You have reached your usage limit for this embedding model. "
                f"Please check your plan and billing details, or wait before trying again."
            )
        elif provider.lower() == "gemini" and "alts creds" in error_lower:
            # Gemini-specific authentication error
            message = (
                f"Google Gemini API authentication failed for model '{model}'. "
                f"The API key may be missing, invalid, or does not have access to this embedding model. "
                f"Please check your GEMINI_API_KEY and ensure it has the necessary permissions."
            )
        else:
            # Generic authentication/permission error
            message = (
                f"API authentication failed for {provider} embedding model '{model}'. "
                f"Please verify the API key has access to this model. Error: {error_str}"
            )

        super().__init__(message)


class EmbeddingError(Exception):
    """Raised when there's a general error computing embeddings."""

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)
