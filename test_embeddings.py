# test_embeddings.py
from src.rag_service.embeddings import create_embeddings_model, similarity_search


def test_embeddings():
    # Create Google embedding model
    google_model = create_embeddings_model("google")

    # Test texts
    text1 = "How do I operate the fish feeding machine?"
    text2 = "What's the proper way to feed fish in the aquarium?"
    text3 = "Tell me about quantum physics."

    # Generate embeddings
    embedding1 = google_model.get_embedding(text1)
    embedding2 = google_model.get_embedding(text2)
    embedding3 = google_model.get_embedding(text3)

    # Calculate similarities
    sim1_2 = similarity_search(embedding1, embedding2)
    sim1_3 = similarity_search(embedding1, embedding3)

    print("Google Embedding Test")
    print(f"Embedding dimensions: {len(embedding1)}")
    print(f"Similarity between related texts: {sim1_2}")
    print(f"Similarity between unrelated texts: {sim1_3}")

    # Compare with OpenAI for reference
    openai_model = create_embeddings_model("openai")
    openai_embedding1 = openai_model.get_embedding(text1)
    openai_embedding2 = openai_model.get_embedding(text2)
    openai_embedding3 = openai_model.get_embedding(text3)

    openai_sim1_2 = similarity_search(openai_embedding1, openai_embedding2)
    openai_sim1_3 = similarity_search(openai_embedding1, openai_embedding3)

    print("\nOpenAI Embedding Test")
    print(f"Embedding dimensions: {len(openai_embedding1)}")
    print(f"Similarity between related texts: {openai_sim1_2}")
    print(f"Similarity between unrelated texts: {openai_sim1_3}")


if __name__ == "__main__":
    test_embeddings()
