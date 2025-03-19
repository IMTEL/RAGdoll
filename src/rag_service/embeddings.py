import math
from abc import ABC, abstractmethod
import openai
import numpy as np
import google.generativeai as genai

from src.config import Config

class EmbeddingsModel(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """
        Get the embedding of a text.

        Args:
            text (str): The text to embed

        Returns:
            list[float]: The embedding of the text
        """
        pass


class OpenAIEmbedding(EmbeddingsModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        
        self.config = Config()
        self.model = self.config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = openai.Client(api_key=self.config.API_KEY)
        self.model_name = model_name

    def get_embedding(self, text: str) -> list[float]:
        
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=text, model=self.model_name, dimensions=768)
        return response.data[0].embedding
    

class GoogleEmbedding(EmbeddingsModel):
    def __init__(self, model_name: str = "text-embedding-004"):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name

    def get_embedding(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        embedding = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=768
            
        )
        return embedding["embedding"]
    

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
        
        
    
    
def similarity_search(embedding1: list[float], embedding2: list[float]) -> float:
    """
    Calculate the cosine similarity between two embedding vectors.
    
    Cosine similarity is defined as:
      cosine_similarity = (A · B) / (||A|| * ||B||)
    
    Args:
        embedding1 (list[float]): The first embedding vector.
        embedding2 (list[float]): The second embedding vector.
    
    Returns:
        float: The cosine similarity between the two embeddings. Returns 0.0 if either vector is zero.
    """
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    
    # Calculate the Euclidean norms (magnitudes) of the vectors
    norm1 = math.sqrt(sum(a * a for a in embedding1))
    norm2 = math.sqrt(sum(b * b for b in embedding2))
    
    # Guard against division by zero if any of the norms is zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Compute cosine similarity
    return dot_product / (norm1 * norm2)


if __name__ == "__main__":
    embeddings_model = create_embeddings_model()
    embedding1 = embeddings_model.get_embedding("Hello, world!")
    embedding2 = embeddings_model.get_embedding("Goodbye, world!")
    similarity = similarity_search(embedding1, embedding2)
    print(f"Similarity: {similarity}")
    
    
    