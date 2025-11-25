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
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send the prompt to the LLM and return the generated response."""


class IdunLLM(LLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        config = Config()
        self.model = model or config.IDUN_MODEL
        self.url = config.IDUN_API_URL
        if not api_key:
            raise LLMAPIError(
                "idun",
                self.model,
                "authentication",
                "IdunLLM requires an explicit api_key. No fallback to config.",
                401,
            )
        self.token = api_key

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

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
            raise LLMGenerationError("idun", self.model, str(e)) from e


class OpenAILLM(LLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = model or self.config.GPT_MODEL
        if not api_key:
            raise LLMAPIError(
                "openai",
                self.model,
                "authentication",
                "OpenAILLM requires an explicit api_key. No fallback to config.",
                401,
            )
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Uses the new API client interface to generate a response."""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except AuthenticationError as e:
            raise LLMAPIError(
                "openai", self.model, "authentication", str(e), 401
            ) from e
        except PermissionDeniedError as e:
            raise LLMAPIError(
                "openai", self.model, "authentication", str(e), 403
            ) from e
        except RateLimitError as e:
            raise LLMAPIError("openai", self.model, "quota", str(e), 429) from e
        except NotFoundError as e:
            raise LLMAPIError(
                "openai", self.model, "model_not_found", str(e), 404
            ) from e
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
                ) from e
            raise LLMGenerationError("openai", self.model, str(e)) from e
        except Exception as e:
            raise LLMGenerationError("openai", self.model, str(e)) from e


class GeminiLLM(LLM):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = model or self.config.GEMINI_MODEL
        if not api_key:
            raise LLMAPIError(
                "gemini",
                self.model,
                "authentication",
                "GeminiLLM requires an explicit api_key. No fallback to config.",
                401,
            )
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Uses the Google Generative AI client to generate a response."""
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
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
                raise LLMAPIError(
                    "gemini", self.model, "authentication", str(e), 401
                ) from e

            # Check for quota/rate limit errors
            if (
                "quota" in error_msg
                or "rate limit" in error_msg
                or "resource exhausted" in error_msg
            ):
                raise LLMAPIError("gemini", self.model, "quota", str(e), 429) from e

            # Check for model not found errors
            if (
                "not found" in error_msg
                or "does not exist" in error_msg
                or "invalid model" in error_msg
            ):
                raise LLMAPIError(
                    "gemini", self.model, "model_not_found", str(e), 404
                ) from e

            # Check for insufficient credits/tokens
            if (
                "insufficient" in error_msg
                or "billing" in error_msg
                or "payment" in error_msg
            ):
                raise LLMAPIError(
                    "gemini", self.model, "insufficient_tokens", str(e), 402
                ) from e

            # Default to generation error
            raise LLMGenerationError("gemini", self.model, str(e)) from e


class MockLLM(LLM):
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return f"Mocked response for prompt: {prompt}"


def _filter_language_models(provider: str, models: list[Model]) -> list[Model]:
    """Filter out non-chat or unsupported models for a provider."""
    blocked_keywords = ["embedding", "moderation", "tts", "whisper", "preview", "audio"]

    if provider == "openai":
        return [
            model
            for model in models
            if model.name.startswith("gpt-")
            and not any(keyword in model.name for keyword in blocked_keywords)
        ]

    if provider in {"gemini", "google"}:
        return [
            model
            for model in models
            if model.name.startswith("models/gemini-")
            and not any(keyword in model.name for keyword in blocked_keywords)
        ]

    return models


def list_idun_models(api_key: str) -> list[Model]:
    config = Config()
    headers = {"Authorization": f"Bearer {api_key}"}

    base_url = config.IDUN_API_URL.rstrip("/")
    if base_url.endswith("/chat/completions"):
        models_url = f"{base_url.rsplit('/', 1)[0]}/models"
    else:
        models_url = f"{base_url}/models"

    try:
        response = requests.get(models_url, headers=headers, timeout=10)
    except Exception as exc:
        raise LLMAPIError("idun", "*", "authentication", str(exc), 502) from exc

    if response.status_code in (401, 403):
        raise LLMAPIError(
            "idun", "*", "authentication", response.text, response.status_code
        )
    if response.status_code == 429:
        raise LLMAPIError("idun", "*", "quota", response.text, response.status_code)
    if response.status_code != 200:
        raise LLMAPIError(
            "idun", "*", "model_not_found", response.text, response.status_code
        )

    data = response.json()
    items = data.get("data", data) if isinstance(data, dict) else data

    models: list[Model] = []
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else None
        if model_id:
            models.append(Model("idun", model_id, True, item.get("description")))

    return models


def list_openai_models(api_key: str) -> list[Model]:
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
    except AuthenticationError as exc:
        raise LLMAPIError("openai", "*", "authentication", str(exc), 401) from exc
    except PermissionDeniedError as exc:
        raise LLMAPIError("openai", "*", "authentication", str(exc), 403) from exc
    except RateLimitError as exc:
        raise LLMAPIError("openai", "*", "quota", str(exc), 429) from exc
    except APIError as exc:
        error_msg = str(exc).lower()
        if (
            "insufficient" in error_msg
            or "quota" in error_msg
            or "billing" in error_msg
        ):
            raise LLMAPIError(
                "openai", "*", "insufficient_tokens", str(exc), 402
            ) from exc
        raise LLMAPIError("openai", "*", "model_not_found", str(exc), 502) from exc
    except Exception as exc:
        raise LLMAPIError("openai", "*", "model_not_found", str(exc), 500) from exc

    results = [
        Model("openai", item.id, False, getattr(item, "description", None))
        for item in models.data
    ]
    return _filter_language_models("openai", results)


def list_gemini_models(api_key: str) -> list[Model]:
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
    except Exception as exc:
        message = str(exc).lower()
        if (
            "api key" in message
            or "authentication" in message
            or "unauthorized" in message
        ):
            raise LLMAPIError("gemini", "*", "authentication", str(exc), 401) from exc
        if (
            "quota" in message
            or "rate limit" in message
            or "resource exhausted" in message
        ):
            raise LLMAPIError("gemini", "*", "quota", str(exc), 429) from exc
        raise LLMAPIError("gemini", "*", "model_not_found", str(exc), 500) from exc

    results = [
        Model("gemini", item.name, False, getattr(item, "description", None))
        for item in models
    ]
    return _filter_language_models("gemini", results)


def list_llm_models(provider: str, api_key: str) -> list[Model]:
    provider_normalized = provider.lower().strip()

    if provider_normalized == "openai":
        models = list_openai_models(api_key)
    elif provider_normalized in {"gemini", "google"}:
        models = list_gemini_models(api_key)
    elif provider_normalized == "idun":
        models = list_idun_models(api_key)
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'")

    return [
        Model(provider_normalized, model.name, model.GDPR_compliant, model.description)
        for model in models
    ]


def create_llm(
    llm_provider: str = "idun", model: str | None = None, api_key: str | None = None
) -> LLM:
    """Factory for creating LLM instances.

    Args:
        llm_provider (str): The LLM service provider ("idun", "openai", "gemini", "mock").
        model (str, optional): The specific model to use. Falls back to config if not provided.
        api_key (str, optional): The API key to use. Falls back to config if not provided.

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    match llm_provider.lower():
        case "idun":
            return IdunLLM(model=model, api_key=api_key)
        case "openai":
            return OpenAILLM(model=model, api_key=api_key)
        case "gemini" | "google":
            return GeminiLLM(model=model, api_key=api_key)
        case "mock":
            return MockLLM()
        case _:
            raise ValueError(f"LLM {llm_provider} not supported")


# if __name__ == "__main__":
#     llm = create_llm("idun")
#     while True:
#         prompt = input("Enter your prompt: ")
#         response = llm.generate(prompt)
#         print(response)
