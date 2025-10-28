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

    @abstractmethod
    def get_available_embedding_models(self) -> list[str]:
        """Get the available embedding models.

        Returns:
            list[str]: The available embedding models
        """


class OpenAIEmbedding(EmbeddingsModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.config = Config()
        self.model = self.config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = openai.Client(api_key=self.config.API_KEY)
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

    def get_available_embedding_models() -> list[str]:
        try:
            client = openai.OpenAI()
            models = client.models.list()
            # Filter for embedding models
            embedding_models = [
                "openai:" + item.id
                for item in models.data
                if "embedding" in item.id.lower()
            ]
            return embedding_models
        except Exception:
            return []


class GoogleEmbedding(EmbeddingsModel):
    def __init__(
        self, model_name: str = "text-embedding-004"
    ):  # TODO: take from agent config
        config = Config()

        genai.configure(api_key=config.GEMINI_API_KEY)  # TODO: take from agent config

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

    def get_available_embedding_models() -> list[str]:
        try:
            models = genai.list_models()
            # Filter for embedding models that support embedContent (not the old embedText API)
            # Exclude models/gemini-embedding-exp (alias that doesn't work consistently)
            embedding_models = []
            for item in models:
                if (
                    "embedding" in item.name.lower()
                    and "embedContent" in item.supported_generation_methods
                    and item.name
                    != "models/gemini-embedding-exp"  # Exclude the alias version
                ):
                    embedding_models.append("gemini:" + item.name)
            return embedding_models
        except Exception:
            return []


def get_available_embedding_models():
    """Get all available embedding models from all providers."""
    openai_models = OpenAIEmbedding.get_available_embedding_models() or []
    google_models = GoogleEmbedding.get_available_embedding_models() or []
    return openai_models + google_models


def create_embeddings_model(
    embeddings_model: str = "gemini:models/text-embedding-004",
) -> EmbeddingsModel:
    """Factory for creating embeddings models.

    Args:
        embeddings_model (str, optional): Embeddings model in format "provider:model_name".
                                          Defaults to "gemini:models/text-embedding-004".
                                          Format differs by provider:
                                          - OpenAI: "openai:text-embedding-3-small"
                                          - Gemini: "gemini:models/text-embedding-004" (includes "models/" prefix)

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
                return OpenAIEmbedding(model_name=model_name)
            return OpenAIEmbedding()
        case "google" | "gemini":
            if model_name:
                return GoogleEmbedding(model_name=model_name)
            return GoogleEmbedding()
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
