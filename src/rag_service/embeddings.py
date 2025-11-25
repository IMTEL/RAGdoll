from abc import ABC, abstractmethod

import google.generativeai as genai
import openai
from numpy import dot
from numpy.linalg import norm

from src.config import Config
from src.models.errors import EmbeddingAPIError, EmbeddingError


class EmbeddingsModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Get the embedding of a text.

        Args:
            text (str): The text to embed

        Returns:
            list[float]: The embedding of the text
        """


class OpenAIEmbedding(EmbeddingsModel):
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        embedding_api_key: str | None = None,
    ):
        self.config = Config()
        self.model = self.config.GPT_MODEL
        if not embedding_api_key:
            raise EmbeddingAPIError(
                "OpenAI",
                model_name,
                "OpenAIEmbedding requires an explicit embedding_api_key.",
            )
        self.client = openai.Client(api_key=embedding_api_key)
        self.model_name = model_name

    def get_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(
                input=text, model=self.model_name, dimensions=768
            )
            return response.data[0].embedding
        except openai.AuthenticationError as e:
            raise EmbeddingAPIError("OpenAI", self.model_name, e) from e
        except openai.PermissionDeniedError as e:
            raise EmbeddingAPIError("OpenAI", self.model_name, e) from e
        except openai.RateLimitError as e:
            raise EmbeddingAPIError("OpenAI", self.model_name, e) from e
        except openai.NotFoundError as e:
            raise EmbeddingError(
                f"OpenAI embedding model '{self.model_name}' not found. "
                "Please verify the model name is correct.",
                e,
            ) from e
        except Exception as e:
            raise EmbeddingError(f"Error getting OpenAI embedding: {e!s}", e) from e


class GoogleEmbedding(EmbeddingsModel):
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        embedding_api_key: str | None = None,
    ):
        Config()
        if not embedding_api_key:
            raise EmbeddingAPIError(
                "Gemini",
                model_name,
                "GoogleEmbedding requires an explicit embedding_api_key.",
            )
        genai.configure(api_key=embedding_api_key)
        self.model_name = model_name
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"

    def get_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        try:
            embedding = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query",
                output_dimensionality=768,
            )
            return embedding["embedding"]
        except Exception as e:
            # Check if it's an authentication or permission error
            error_msg = str(e).lower()
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "unauthorized" in error_msg
                or "forbidden" in error_msg
                or "alts creds" in error_msg  # Google Cloud authentication error
                or "not running on gcp" in error_msg
                or "quota exceeded" in error_msg  # Rate limit / quota errors
                or "rate limit" in error_msg
                or "429" in error_msg  # HTTP 429 Too Many Requests
            ):
                raise EmbeddingAPIError("Gemini", self.model_name, e) from e
            elif (
                "not found" in error_msg
                or "invalid model" in error_msg
                or "does not support" in error_msg  # Method not supported
                or "embedtext" in error_msg  # Old API method
            ):
                raise EmbeddingError(
                    f"Gemini embedding model '{self.model_name}' not found or incompatible. "
                    "Please verify the model name is correct and supports the embedContent API.",
                    e,
                ) from e
            else:
                raise EmbeddingError(f"Error getting Gemini embedding: {e!s}", e) from e


def _filter_openai_embedding_models(models, provider_label: str) -> list[str]:
    filtered: list[str] = []
    for item in models:
        model_id = getattr(item, "id", None)
        if not isinstance(model_id, str):
            continue
        if "embedding" in model_id.lower():
            filtered.append(f"{provider_label}:{model_id}")
    return filtered


def list_openai_embedding_models(
    api_key: str, provider_label: str = "openai"
) -> list[str]:
    try:
        client = openai.Client(api_key=api_key)
        response = client.models.list()
    except openai.AuthenticationError as exc:
        raise EmbeddingAPIError("OpenAI", "*", exc) from exc
    except openai.PermissionDeniedError as exc:
        raise EmbeddingAPIError("OpenAI", "*", exc) from exc
    except openai.RateLimitError as exc:
        raise EmbeddingAPIError("OpenAI", "*", exc) from exc
    except Exception as exc:
        raise EmbeddingError("Failed to list OpenAI embedding models", exc) from exc

    return _filter_openai_embedding_models(response.data, provider_label)


def list_gemini_embedding_models(
    api_key: str, provider_label: str = "gemini"
) -> list[str]:
    try:
        genai.configure(api_key=api_key)
        models = list(genai.list_models())
    except Exception as exc:
        message = str(exc).lower()
        if (
            "api key" in message
            or "authentication" in message
            or "permission" in message
            or "unauthorized" in message
            or "forbidden" in message
        ):
            raise EmbeddingAPIError("Gemini", "*", exc) from exc
        if "quota" in message or "rate limit" in message or "429" in message:
            raise EmbeddingAPIError("Gemini", "*", exc) from exc
        raise EmbeddingError("Failed to list Gemini embedding models", exc) from exc

    embedding_models: list[str] = []
    for item in models:
        name = getattr(item, "name", "")
        methods = getattr(item, "supported_generation_methods", []) or []
        if not isinstance(name, str) or not name:
            continue
        name_lower = name.lower()
        methods_lower = [method.lower() for method in methods]
        if (
            "embedding" in name_lower
            and "embedcontent" in methods_lower
            and name != "models/gemini-embedding-exp"
        ):
            embedding_models.append(f"{provider_label}:{name}")

    return embedding_models


def list_embedding_models(provider: str, api_key: str) -> list[str]:
    normalized = provider.lower().strip()

    if normalized == "openai":
        return list_openai_embedding_models(api_key, provider_label=normalized)
    if normalized in {"gemini", "google"}:
        return list_gemini_embedding_models(api_key, provider_label=normalized)

    raise ValueError(
        f"Embedding provider '{provider}' not supported. Use 'openai' or 'gemini'."
    )


def create_embeddings_model(
    embeddings_model: str = "gemini:models/text-embedding-004",
    embedding_api_key: str | None = None,
) -> EmbeddingsModel:
    """Factory for creating embeddings models.

    Args:
        embeddings_model (str, optional): Embeddings model in format "provider:model_name".
            Defaults to "gemini:models/text-embedding-004".
            Format differs by provider:
            - OpenAI: "openai:text-embedding-3-small"
            - Gemini: "gemini:models/text-embedding-004" (includes "models/" prefix)
        embedding_api_key (str | None, optional): API key for the embedding model. Defaults to None.

    Examples:
        - "openai:text-embedding-3-small"
        - "openai:text-embedding-3-large"
        - "gemini:models/text-embedding-004"
        - "gemini:models/embedding-001"

    Raises:
        ValueError: If provider is not supported or format is invalid

    Returns:
        EmbeddingsModel: Instance of the appropriate embeddings model
    """
    # Parse provider:model_name format
    if ":" in embeddings_model:
        provider, model_name = embeddings_model.split(":", 1)
        provider = provider.lower().strip()
        model_name = model_name.strip()
    else:
        raise ValueError(
            f"Embedding '{embeddings_model}' not supported. "
            "Format must be 'provider:model_name' (e.g., 'openai:text-embedding-3-small' or 'gemini:models/text-embedding-004')"
        )

    match provider.lower():
        case "openai":
            if model_name:
                return OpenAIEmbedding(
                    model_name=model_name, embedding_api_key=embedding_api_key
                )
            return OpenAIEmbedding(embedding_api_key=embedding_api_key)
        case "google" | "gemini":
            if model_name:
                return GoogleEmbedding(
                    model_name=model_name, embedding_api_key=embedding_api_key
                )
            return GoogleEmbedding(embedding_api_key=embedding_api_key)
        case _:
            raise ValueError(
                f"Embedding provider '{provider}' not supported. Use 'openai' or 'gemini' in the form 'provider:model_name'."
            )


def similarity_search(list_1, list_2):
    product = norm(list_1) * norm(list_2)
    if product == 0:
        return 0.0
    cos_sim = dot(list_1, list_2) / product
    return float(cos_sim)


if __name__ == "__main__":
    embeddings_model = create_embeddings_model()
    embedding1 = embeddings_model.get_embedding("Hello, world!")
    embedding2 = embeddings_model.get_embedding("Goodbye, world!")
    similarity = similarity_search(embedding1, embedding2)
    print(f"Similarity: {similarity}")
