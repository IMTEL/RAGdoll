from typing import Protocol

import google.generativeai as genai
import requests
from openai import (
    APIError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src.config import Config
from src.models.errors import LLMAPIError, LLMGenerationError
from src.models.model import Model


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        """Send the prompt to the LLM and return the generated response."""

    @staticmethod
    def get_models() -> list[str]:
        """Returns the models which can be used by the provider."""


class IdunLLM(LLM):
    def __init__(self, model: str | None = None):
        config = Config()
        self.model = model or config.IDUN_MODEL
        self.url = config.IDUN_API_URL
        self.token = config.IDUN_API_KEY

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}

        try:
            response = requests.post(self.url, headers=headers, json=data)

            # Handle HTTP errors
            if response.status_code == 401 or response.status_code == 403:
                raise LLMAPIError(
                    "idun",
                    self.model,
                    "authentication",
                    f"HTTP {response.status_code}: {response.text}",
                    response.status_code,
                )

            if response.status_code == 429:
                raise LLMAPIError(
                    "idun",
                    self.model,
                    "quota",
                    f"HTTP {response.status_code}: {response.text}",
                    429,
                )

            if response.status_code == 404:
                raise LLMAPIError(
                    "idun",
                    self.model,
                    "model_not_found",
                    f"HTTP {response.status_code}: {response.text}",
                    404,
                )

            if response.status_code != 200:
                error_text = response.text.lower()
                if (
                    "insufficient" in error_text
                    or "quota" in error_text
                    or "billing" in error_text
                ):
                    raise LLMAPIError(
                        "idun",
                        self.model,
                        "insufficient_tokens",
                        f"HTTP {response.status_code}: {response.text}",
                        402,
                    )
                raise LLMGenerationError(
                    "idun", self.model, f"HTTP {response.status_code}: {response.text}"
                )

            return response.json()["choices"][0]["message"]["content"].strip()

        except LLMAPIError:
            # Re-raise API errors
            raise
        except LLMGenerationError:
            # Re-raise generation errors
            raise
        except Exception as e:
            # Catch any other errors (network issues, JSON parsing, etc.)
            raise LLMGenerationError("idun", self.model, str(e))

    @staticmethod
    def get_models() -> list[str]:
        try:
            models = Config().IDUN_MODELS
            if len(models) == 0:
                return []
            return [Model("idun", name, True, None) for name in models]
        except Exception:
            return []


class OpenAILLM(LLM):
    def __init__(self, model: str | None = None):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = model or self.config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = OpenAI(api_key=self.config.API_KEY)

    def generate(self, prompt: str) -> str:
        """Uses the new API client interface to generate a response."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )
            return response.choices[0].message.content.strip()
        except AuthenticationError as e:
            raise LLMAPIError("openai", self.model, "authentication", str(e), 401)
        except PermissionDeniedError as e:
            raise LLMAPIError("openai", self.model, "authentication", str(e), 403)
        except RateLimitError as e:
            raise LLMAPIError("openai", self.model, "quota", str(e), 429)
        except NotFoundError as e:
            raise LLMAPIError("openai", self.model, "model_not_found", str(e), 404)
        except APIError as e:
            # Check if it's a token/credit issue
            error_msg = str(e).lower()
            if (
                "insufficient" in error_msg
                or "quota" in error_msg
                or "billing" in error_msg
            ):
                raise LLMAPIError(
                    "openai", self.model, "insufficient_tokens", str(e), 402
                )
            raise LLMGenerationError("openai", self.model, str(e))
        except Exception as e:
            raise LLMGenerationError("openai", self.model, str(e))

    @staticmethod
    def get_models() -> list[str]:
        try:
            client = OpenAI()
            models = client.models.list()
            return [Model("openai", item.id, False, None) for item in models.data]
        except Exception:
            return []


class GeminiLLM(LLM):
    def __init__(self, model: str | None = None):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = model or self.config.GEMINI_MODEL

        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> str:
        """Uses the Google Generative AI client to generate a response."""
        try:
            response = self.client.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e).lower()

            # Check for authentication/API key errors
            if (
                "api key" in error_msg
                or "api_key" in error_msg
                or "unauthorized" in error_msg
                or "authentication" in error_msg
            ):
                raise LLMAPIError("gemini", self.model, "authentication", str(e), 401)

            # Check for quota/rate limit errors
            if (
                "quota" in error_msg
                or "rate limit" in error_msg
                or "resource exhausted" in error_msg
            ):
                raise LLMAPIError("gemini", self.model, "quota", str(e), 429)

            # Check for model not found errors
            if (
                "not found" in error_msg
                or "does not exist" in error_msg
                or "invalid model" in error_msg
            ):
                raise LLMAPIError("gemini", self.model, "model_not_found", str(e), 404)

            # Check for insufficient credits/tokens
            if (
                "insufficient" in error_msg
                or "billing" in error_msg
                or "payment" in error_msg
            ):
                raise LLMAPIError(
                    "gemini", self.model, "insufficient_tokens", str(e), 402
                )

            # Default to generation error
            raise LLMGenerationError("gemini", self.model, str(e))

    @staticmethod
    def get_models() -> list[str]:
        try:
            models = genai.list_models()
            print(models)
            return [Model("gemini", item.name, False, None) for item in models]
        except Exception:
            return []


class MockLLM(LLM):
    def generate(self, prompt: str) -> str:
        return f"Mocked response for prompt: {prompt}"

    @staticmethod
    def get_models() -> list[str]:
        return []


def get_models():
    all_models = OpenAILLM.get_models() + GeminiLLM.get_models()

    model_filter = ["embedding", "moderation", "tts", "whisper", "preview", "audio"]

    language_models = [
        model
        for model in all_models
        if model.name.startswith(("gpt-", "models/gemini-"))
        and not any(kw in model.name for kw in model_filter)
    ]

    return language_models + IdunLLM.get_models()


def create_llm(llm_provider: str = "idun", model: str | None = None) -> LLM:
    """Factory for creating LLM instances.

    Args:
        llm_provider (str): The LLM service provider ("idun", "openai", "gemini", "mock").
        model (str, optional): The specific model to use. Falls back to config if not provided.

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    match llm_provider.lower():
        case "idun":
            return IdunLLM(model=model)
        case "openai":
            return OpenAILLM(model=model)
        case "gemini":
            return GeminiLLM(model=model)
        case "mock":
            return MockLLM()
        case _:
            raise ValueError(f"LLM {llm_provider} not supported")


if __name__ == "__main__":
    llm = create_llm("idun")
    while True:
        prompt = input("Enter your prompt: ")
        response = llm.generate(prompt)
        print(response)
