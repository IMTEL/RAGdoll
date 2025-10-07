import time
from uuid import uuid4

import pytest

from src.config import Config
from src.rag_service.context import Context
from src.rag_service.embeddings import (
    create_embeddings_model,
    similarity_search,
)
from src.rag_service.dao import get_context_dao


@pytest.mark.integration
def test_database_is_reachable():
    """Test the is_reachable method to ensure we can connect to MongoDB or use the mock DB."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_context_dao()
    assert db.is_reachable() is True, "Database should be reachable"


@pytest.mark.integration
def test_post_context_and_retrieve_by_npc():
    """Test post_context, then verify that get_context_from_npc can retrieve the inserted document."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_context_dao()
    test_text = "Test text for category"
    test_document_name = "TestDocCategory!"
    test_category = "test_category_123"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.insert_context(
        document_id=test_id,
        embedding=test_embedding,
        context=Context(
            text=test_text,
            document_name=test_document_name,
            category=test_category,
        ),
    )
    time.sleep(1)
    assert post_result is True, "post_context should return True"

    # Retrieve the context by category
    retrieved_contexts = db.get_context_by_category(test_category)
    assert len(retrieved_contexts) > 0, "Should retrieve at least one context"
    # Check that the first context matches what we posted
    context = retrieved_contexts[0]
    assert context.text == test_text, "Text should match the posted text"
    assert context.document_name == test_document_name, "Document name should match"
    assert context.category == test_category, "Category should match"


@pytest.mark.integration
def test_post_context_and_retrieve_by_embedding():
    """Test post_context, then verify get_context returns the document when the similarity is above the threshold."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_context_dao()
    test_text = "Embedding-based retrieval text"
    test_document_name = "EmbeddingDoc"
    test_category = "test_category"
    test_embedding = [0.1] * 768
    test_id = str(uuid4())

    # Post the context
    post_result = db.insert_context(
        document_id=test_id,
        embedding=test_embedding,
        context=Context(
            text=test_text,
            document_name=test_document_name,
            category=test_category,
        ),
    )
    time.sleep(1)
    assert post_result is True, "post_context should return True"

    # Retrieve context by (document_id, embedding)
    retrieved_contexts = db.get_context(test_id, test_embedding)
    assert len(retrieved_contexts) > 0, "Should retrieve at least one context"
    context = retrieved_contexts[0]
    assert context.text == test_text, "Text should match the posted text"
    assert context.document_name == test_document_name, "Document name should match"
    assert context.category == test_category, "Category should match"


@pytest.mark.integration
def test_get_context_from_npc_no_results():
    """Test get_context_by_category with a category that doesn't exist to confirm it raises a ValueError."""
    if Config().ENV != "dev":
        pytest.skip("Skipping test that requires MongoDB for mock DB")
    db = get_context_dao()
    non_existent_category = "NonExistentCategory999999"

    with pytest.raises(ValueError) as exc_info:
        db.get_context_by_category(non_existent_category)

    assert f"No documents found for category: {non_existent_category}" in str(
        exc_info.value
    ), "Should raise ValueError if category not found"


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
