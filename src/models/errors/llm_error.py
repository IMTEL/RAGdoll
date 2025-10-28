"""LLM-specific error classes for better error handling and user feedback."""


class LLMError(Exception):
    """Base class for LLM-related errors."""

    def __init__(self, message: str, provider: str, model: str, status_code: int = 500):
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        super().__init__(self.message)


class LLMAPIError(LLMError):
    """Raised when there are API-related issues (auth, quota, model access, etc.)."""

    def __init__(
        self,
        provider: str,
        model: str,
        error_type: str,
        original_error: str = "",
        status_code: int = 401,
    ):
        """Initialize LLMAPIError.

        Args:
        provider: The LLM provider (openai, gemini, idun)
        model: The model name
        error_type: Type of API error (authentication, quota, model_not_found, insufficient_tokens)
        original_error: The original error message
        status_code: HTTP status code to return.
        """
        error_messages = {
            "authentication": (
                f"Authentication failed for {provider} model '{model}'. "
                f"Please verify that:\n"
                f"1. Your {provider.upper()} API key is valid and active\n"
                f"2. The API key has access to the '{model}' model\n"
                f"3. Your account has the necessary permissions"
            ),
            "quota": (
                f"Quota or rate limit exceeded for {provider} model '{model}'. "
                f"Please:\n"
                f"1. Check your {provider.upper()} account quota/billing\n"
                f"2. Wait a few moments and try again\n"
                f"3. Consider upgrading your {provider.upper()} plan if needed"
            ),
            "model_not_found": (
                f"Model '{model}' not found or not accessible for {provider}. "
                f"Please verify that:\n"
                f"1. The model name is correct\n"
                f"2. The model is available in your region\n"
                f"3. Your API key has access to this model"
            ),
            "insufficient_tokens": (
                f"Insufficient tokens or credits for {provider} model '{model}'. "
                f"Please:\n"
                f"1. Check your {provider.upper()} account balance\n"
                f"2. Add credits or upgrade your plan\n"
                f"3. Verify your billing information is up to date"
            ),
        }

        message = error_messages.get(
            error_type, f"API error for {provider} model '{model}'"
        )
        if original_error:
            message += f"\n\nDetails: {original_error}"

        self.error_type = error_type
        super().__init__(message, provider, model, status_code)


class LLMGenerationError(LLMError):
    """Raised when LLM generation/response fails."""

    def __init__(self, provider: str, model: str, original_error: str = ""):
        message = (
            f"Failed to generate response from {provider} model '{model}'. "
            f"This could be due to:\n"
            f"1. Temporary service issues\n"
            f"2. Invalid request parameters\n"
            f"3. Model-specific constraints\n"
            f"4. Network connectivity problems"
        )
        if original_error:
            message += f"\n\nDetails: {original_error}"
        super().__init__(message, provider, model, status_code=503)
