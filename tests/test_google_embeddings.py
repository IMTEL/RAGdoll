import pytest
from src.rag_service.embeddings import create_embeddings_model, GoogleEmbedding, similarity_search

@pytest.mark.integration
def test_create_embeddings_model_google():
    """
    Test that you can create and recieve a GoogleEmbedding, which implements EmbeddingsModel.
    """
    model = create_embeddings_model("google")
    assert isinstance(model, GoogleEmbedding), "Should be a GoogleEmbedding instance"

@pytest.mark.integration
def test_google_embedding_get_embedding():
    """
    Test that the GoogleEmbedding model can generate embeddings.
    """
    model = create_embeddings_model("google")
    test_text = "This is a test sentence."

    embedding = model.get_embedding(test_text)

    # Basic validation of the embedding
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    assert all(isinstance(x, float) for x in embedding), "All elements should be floats"

@pytest.mark.integration
def test_google_embedding_similarity():
    """
    Test that similarity works with Google embeddings.
    """
    model = create_embeddings_model("google")

    # Generate embedding for two similar texts
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "A fast brown fox leaps over a sleeping dog."

    embedding1 = model.get_embedding(text1)
    embedding2 = model.get_embedding(text2)
    
    similarity = similarity_search(embedding1, embedding2)
    
    # Similar sentences should have reasonable similarity
    assert similarity > 0.6, "Similar sentences should have higher similarity"
    
    # Test dissimilar sentences
    text3 = "Quantum physics describes the behavior of subatomic particles."
    embedding3 = model.get_embedding(text3)
    
    similarity_different = similarity_search(embedding1, embedding3)
    
    # Different topics should have lower similarity
    assert similarity_different < similarity, "Different topics should have lower similarity"

@pytest.mark.integration
def test_comparison_between_embedding_providers():
    """
    Compare results from both OpenAI and Google embedding models.
    """
    openai_model = create_embeddings_model("openai")
    google_model = create_embeddings_model("google")
    
    test_text = "This is a test sentence to compare embeddings."
    
    openai_embedding = openai_model.get_embedding(test_text)
    google_embedding = google_model.get_embedding(test_text)
    
    # Verify both return valid embeddings
    assert len(openai_embedding) > 0, "OpenAI embedding should not be empty"
    assert len(google_embedding) > 0, "Google embedding should not be empty"
    
    # Output embedding dimensions for comparison
    print(f"\nOpenAI embedding dimensions: {len(openai_embedding)}")
    print(f"Google embedding dimensions: {len(google_embedding)}")