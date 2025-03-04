import time
import pytest
from uuid import uuid4
import sys
import os


from src.rag_service.dao import get_database
from src.rag_service.context import Context


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
    assert context.NPC == test_NPC, "NPC should match"

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
    assert context.NPC == test_NPC, "NPC should match"

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
