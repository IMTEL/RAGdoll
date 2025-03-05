import time
import pytest
from uuid import uuid4
import sys
import os
from typing import Any


from src.rag_service.dao import get_database
from src.rag_service.context import Context
from src.rag_service.embeddings import (
    create_embeddings_model,
    similarity_search,
    EmbeddingsModel,
    OpenAIEmbedding,
)
from src.LLM import OpenAI_LLM, create_llm

@pytest.mark.integration
def test_database_is_reachable():
    """
    Test the is_reachable method to ensure
    we can connect to MongoDB or use the mock DB.
    """
    db = get_database()
    assert db.is_reachable() is True, "Database should be reachable"

@pytest.mark.integration
def test_post_context_and_retrieve_by_NPC():
    """
    Test post_context, then verify that get_context_from_NPC
    can retrieve the inserted document.
    """
    db = get_database()
    test_text = "Test text for NPC"
    test_document_name = "TestDocNPC!"
    test_NPC = "123"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.post_context(
        text=test_text,
        document_name=test_document_name,
        embedding=test_embedding,
        NPC=test_NPC,
        document_id=test_id,
    )
    time.sleep(1)
    assert post_result is True, "post_context should return True"


    # Retrieve the context by NPC
    retrieved_contexts = db.get_context_from_NPC(test_NPC)
    assert len(retrieved_contexts) > 0, "Should retrieve at least one context"
    # Check that the first context matches what we posted
    context = retrieved_contexts[0]
    assert context.text == test_text, "Text should match the posted text"
    assert context.document_name == test_document_name, "Document name should match"
    assert str(context.NPC) == test_NPC, "NPC should match"

@pytest.mark.integration
def test_post_context_and_retrieve_by_embedding():
    """
    Test post_context, then verify get_context returns the document
    when the similarity is above the threshold.
    """
    db = get_database()
    test_text = "Embedding-based retrieval text"
    test_document_name = "EmbeddingDoc"
    test_NPC = "999"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.post_context(
        text=test_text,
        document_name=test_document_name,
        NPC=test_NPC,
        embedding=test_embedding,
        document_id=test_id,
    )
    time.sleep(1)
    assert post_result is True, "post_context should return True"

    # Retrieve context by (documentId, embedding)
    retrieved_contexts = db.get_context(test_id, test_embedding)
    assert len(retrieved_contexts) > 0, "Should retrieve at least one context"
    context = retrieved_contexts[0]
    assert context.text == test_text, "Text should match the posted text"
    assert context.document_name == test_document_name, "Document name should match"
    assert str(context.NPC) == str(test_NPC), "NPC should match"

@pytest.mark.integration
def test_get_context_from_NPC_no_results():
    """
    Test get_context_from_NPC with an NPC that doesn't exist
    to confirm it raises a ValueError (as per your code).
    """
    db = get_database()
    non_existent_NPC = "999999"

    with pytest.raises(ValueError) as exc_info:
        db.get_context_from_NPC(non_existent_NPC)

    assert f"No documents found for NPC: {non_existent_NPC}" in str(exc_info.value), \
        "Should raise ValueError if NPC not found"


@pytest.mark.integration
def test_create_embeddings_model_openai():
    """
    Test that create_embeddings_model('openai') returns an instance
    of OpenAIEmbedding, which implements EmbeddingsModel.
    """
    model = create_embeddings_model("openai")
    assert isinstance(model, EmbeddingsModel), "Should return a subclass of EmbeddingsModel"
    assert isinstance(model, OpenAIEmbedding), "Should specifically be OpenAIEmbedding"


@pytest.mark.integration
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
    model = create_embeddings_model("openai")
    test_text = "Hello, world!"

    embedding = model.get_embedding(test_text)

    # Basic checks on the embedding
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    assert all(isinstance(x, float) for x in embedding), "All embedding values should be floats"


@pytest.mark.integration
def test_similarity_search_identical_texts():
    """
    Test that two identical texts produce embeddings with a high similarity score.
    """
    model = create_embeddings_model("openai")

    text = "OpenAI is powering the next generation of AI applications."
    embedding1 = model.get_embedding(text)
    embedding2 = model.get_embedding(text)

    similarity = similarity_search(embedding1, embedding2)

    # For identical texts, we expect a similarity very close to 1.0
    assert similarity > 0.9, f"Similarity for identical text was too low: {similarity}"


@pytest.mark.integration
def test_similarity_search_different_texts():
    """
    Test that two very different texts produce embeddings
    with a lower similarity score (though exact threshold may vary).
    """
    model = create_embeddings_model("openai")

    embedding1 = model.get_embedding("The Eiffel Tower is in Paris.")
    embedding2 = model.get_embedding("Quantum physics deals with subatomic particles.")

    similarity = similarity_search(embedding1, embedding2)

    # We just check that it's significantly less than 1.0
    # The exact number can vary, but let's pick a reasonable upper bound
    # for dissimilar sentences:
    assert similarity < 0.8, f"Similarity for very different texts was unexpectedly high: {similarity}"


@pytest.mark.integration
def test_similarity_search_zero_vector():
    """
    Test that similarity_search returns 0.0 if either embedding is all zeros.
    """
    zero_vector = [0.0] * 768  # typical embedding size for OpenAI
    nonzero_vector = [0.1] * 768

    similarity = similarity_search(zero_vector, nonzero_vector)
    assert similarity == 0.0, "Similarity should be 0 if one vector is zero"
    

# === Test LLM ===

class FakeChoice:
    def __init__(self, content: str):
        # Create a fake message object with a 'content' attribute.
        self.message = type("FakeMessage", (), {"content": content})

class FakeResponse:
    def __init__(self, content: str):
        # Simulate a response with a list of choices.
        self.choices = [FakeChoice(content)]


def fake_completions_create(*, model: str, messages: Any) -> FakeResponse:
    """
    Fake function to simulate OpenAI API call.
    It ignores the inputs and returns a fake response.
    """
    return FakeResponse("Test response")


# --- Tests ---
@pytest.mark.integration
def test_create_prompt():
    """
    Test that create_prompt correctly appends additional context.
    """
    # Instantiate an LLM instance. (No API call is made here.)
    llm = OpenAI_LLM()
    base_prompt = "Explain the significance of Python."
    # Provide some extra context as keyword arguments.
    prompt = llm.create_prompt(base_prompt, audience="beginner", detail="basic overview")
    expected_prompt = "Explain the significance of Python.\naudience: beginner\ndetail: basic overview"
    assert prompt == expected_prompt, "create_prompt should combine the base prompt with additional context"


@pytest.fixture
def patched_llm(monkeypatch):
    """
    Fixture to return an OpenAI_LLM instance with the API call patched to avoid spending tokens.
    """
    llm = OpenAI_LLM()
    # Patch the chat.completions.create method so that it returns our fake response.
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_completions_create)
    return llm


@pytest.mark.integration
def test_generate_with_patched_llm(patched_llm):
    """
    Test that generate returns the fake response from the patched API call.
    """
    test_prompt = "Dummy prompt"
    response = patched_llm.generate(test_prompt)
    assert response == "Test response", "generate should return the fake response content"


@pytest.mark.integration
def test_create_llm_valid():
    """
    Test that create_llm returns an instance of OpenAI_LLM when provided 'openai'.
    """
    llm_instance = create_llm("openai")
    assert isinstance(llm_instance, OpenAI_LLM), "create_llm('openai') should return an OpenAI_LLM instance"


@pytest.mark.integration
def test_create_llm_invalid():
    """
    Test that create_llm raises a ValueError when an unsupported LLM is specified.
    """
    with pytest.raises(ValueError) as exc_info:
        create_llm("unsupported")
    assert "not supported" in str(exc_info.value), "create_llm should raise ValueError for unsupported LLMs"