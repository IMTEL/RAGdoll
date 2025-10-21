from abc import ABC, abstractmethod

import google.generativeai as genai
import openai
from numpy import dot
from numpy.linalg import norm

from src.config import Config


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
        response = self.client.embeddings.create(
            input=text, model=self.model_name, dimensions=768
        )
        return response.data[0].embedding

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
        embedding = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768,
        )
        return embedding["embedding"]

    def get_available_embedding_models() -> list[str]:
        try:
            models = genai.list_models()
            # Filter for embedding models
            embedding_models = [
                "gemini:" + item.name
                for item in models
                if "embedding" in item.name.lower()
            ]
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
