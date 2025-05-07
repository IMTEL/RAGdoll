import time
import pytest
from uuid import uuid4
import sys
import os
from typing import Any

from src.config import Config
from src.rag_service.dao import get_database
from src.rag_service.context import Context
from src.rag_service.embeddings import (
    create_embeddings_model,
    similarity_search,
    EmbeddingsModel,
    OpenAIEmbedding,
    GoogleEmbedding,
)
from src.LLM import OpenAI_LLM, create_llm


@pytest.mark.unit
def test_create_embeddings_model_openai():
    """
    Test that create_embeddings_model('openai') returns an instance
    of OpenAIEmbedding, which implements EmbeddingsModel.
    """
    model = create_embeddings_model("openai")
    assert isinstance(model, EmbeddingsModel), "Should return a subclass of EmbeddingsModel"
    assert isinstance(model, OpenAIEmbedding), "Should specifically be OpenAIEmbedding"


@pytest.mark.unit
def test_create_embeddings_model_unsupported_raises():
    """
    Test that create_embeddings_model with an unsupported string
    raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        create_embeddings_model("some-unsupported-model")

    assert "not supported" in str(exc_info.value), "Should raise ValueError for unsupported model"


@pytest.mark.integration
def test_openai_get_embedding():
    """
    Test that we can call get_embedding on a piece of text
    and receive a list of floats with nonzero length.
    """
    model = create_embeddings_model()
    test_text = "Hello, world!"

    embedding = model.get_embedding(test_text)

    # Basic checks on the embedding
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    assert all(isinstance(x, float) for x in embedding), "All embedding values should be floats"


@pytest.mark.unit
def test_similarity_search_identical_texts():
    """
    Test that two identical texts produce embeddings with a high similarity score.
    """
    model = create_embeddings_model()

    text = "OpenAI is powering the next generation of AI applications."
    embedding1 = model.get_embedding(text)
    embedding2 = model.get_embedding(text)

    similarity = similarity_search(embedding1, embedding2)

    # For identical texts, we expect a similarity very close to 1.0
    assert similarity > 0.9, f"Similarity for identical text was too low: {similarity}"


@pytest.mark.unit
def test_similarity_search_different_texts():
    """
    Test that two very different texts produce embeddings
    with a lower similarity score (though exact threshold may vary).
    """
    model = create_embeddings_model()

    embedding1 = model.get_embedding("The Eiffel Tower is in Paris.")
    embedding2 = model.get_embedding("Quantum physics deals with subatomic particles.")

    similarity = similarity_search(embedding1, embedding2)

    # We just check that it's significantly less than 1.0
    # The exact number can vary, but let's pick a reasonable upper bound
    # for dissimilar sentences:
    assert similarity < 0.8, f"Similarity for very different texts was unexpectedly high: {similarity}"


@pytest.mark.unit
def test_similarity_search_zero_vector():
    """
    Test that similarity_search returns 0.0 if either embedding is all zeros.
    """
    zero_vector = [0.0] * 768  # typical embedding size for OpenAI
    nonzero_vector = [0.1] * 768

    similarity = similarity_search(zero_vector, nonzero_vector)
    assert similarity == 0.0, "Similarity should be 0 if one vector is zero"
    

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
