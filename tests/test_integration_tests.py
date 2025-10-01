import time
from uuid import uuid4

import pytest

from src.config import Config
from src.rag_service.dao import get_database
from src.rag_service.embeddings import (
    create_embeddings_model,
    similarity_search,
)


@pytest.mark.integration
def test_database_is_reachable():
    """Test the is_reachable method to ensure we can connect to MongoDB or use the mock DB."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_database()
    assert db.is_reachable() is True, "Database should be reachable"


@pytest.mark.integration
def test_post_context_and_retrieve_by_npc():
    """Test post_context, then verify that get_context_from_npc can retrieve the inserted document."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_database()
    test_text = "Test text for NPC"
    test_document_name = "TestDocNPC!"
    test_npc = "123"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.post_context(
        text=test_text,
        document_name=test_document_name,
        embedding=test_embedding,
        NPC=test_npc,
        document_id=test_id,
    )
    time.sleep(1)
    assert post_result is True, "post_context should return True"

    # Retrieve the context by NPC
    retrieved_contexts = db.get_context_from_npc(test_npc)
    assert len(retrieved_contexts) > 0, "Should retrieve at least one context"
    # Check that the first context matches what we posted
    context = retrieved_contexts[0]
    assert context.text == test_text, "Text should match the posted text"
    assert context.document_name == test_document_name, "Document name should match"
    assert str(context.NPC) == test_npc, "NPC should match"


@pytest.mark.integration
def test_post_context_and_retrieve_by_embedding():
    """Test post_context, then verify get_context returns the document when the similarity is above the threshold."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_database()
    test_text = "Embedding-based retrieval text"
    test_document_name = "EmbeddingDoc"
    test_npc = "999"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.post_context(
        text=test_text,
        document_name=test_document_name,
        NPC=test_npc,
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
    assert str(context.NPC) == str(test_npc), "NPC should match"


@pytest.mark.integration
def test_get_context_from_npc_no_results():
    """Test get_context_from_npc with an NPC that doesn't exist to confirm it raises a ValueError (as per your code)."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_database()
    non_existent_npc = "999999"

    with pytest.raises(ValueError) as exc_info:
        db.get_context_from_npc(non_existent_npc)

    assert f"No documents found for NPC: {non_existent_npc}" in str(exc_info.value), (
        "Should raise ValueError if NPC not found"
    )


@pytest.mark.integration
def test_comparison_between_embedding_providers():
    """Compare results from both OpenAI and Google embedding models."""
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


@pytest.mark.integration
def test_google_embedding_similarity():
    """Test that similarity works with Google embeddings."""
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
    assert similarity_different < similarity, (
        "Different topics should have lower similarity"
    )
