
import math

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
