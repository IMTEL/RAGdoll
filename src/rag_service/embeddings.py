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
    return (
        OpenAIEmbedding.get_available_embedding_models()
        + GoogleEmbedding.get_available_embedding_models()
    )


# TODO: Remake this to use with an agent config
def create_embeddings_model(embeddings_model: str = "google") -> EmbeddingsModel:
    """Factory for creating embeddings models.

    Args:
        embeddings_model (str, optional): Select embeddings model. Defaults to "openai".

    Raises:
        ValueError: _description_

    Returns:
        EmbeddingsModel: _description_
    """
    match embeddings_model.lower():
        case "openai":
            return OpenAIEmbedding()
        case "google":
            return GoogleEmbedding()
        case _:
            raise ValueError(f"Embeddings model {embeddings_model} not supported")


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
