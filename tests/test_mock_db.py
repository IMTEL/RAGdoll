import os
import pytest
from src.rag_service.database.mock_database_client import (
    upload_document,
    list_documents,
    get_document_by_id,
    get_document_by_name,
)
from src.config import Config  # Import Config

TEST_FILE = "tests/test_sets/test_document.txt"
TEST_NPC = 100

@pytest.fixture(scope="module", autouse=True)
def mock_db():
    """Set up the mock database environment for the test session and skip if not using mock."""
    config = Config()
    if config.RAG_DATABASE_SYSTEM != "mock":
        pytest.skip("Skipping tests because RAG_DATABASE_SYSTEM is not set to mock")
    return None  # Return None as the fixture isn't actually used in the tests

def test_upload_and_list():
    """Test uploading a document and then listing all documents."""
    # No need to check config here as the fixture will skip the test if needed
    initial_documents = list_documents()
    upload_success = upload_document(TEST_FILE, TEST_NPC)
    assert upload_success, f"Failed to upload {TEST_FILE}"

    documents = list_documents()
    assert len(documents) == len(initial_documents) + 1, "List should contain one more document after upload"

    uploaded_doc = documents[-1]
    assert uploaded_doc["document_name"] == os.path.basename(TEST_FILE), "Document name mismatch"
    assert int(uploaded_doc["npc"]) == TEST_NPC, "NPC mismatch"

def test_upload_and_get_by_id():
    """Test uploading a document and then retrieving it by ID."""
    # No need to check config here as the fixture will skip the test if needed
    upload_success = upload_document(TEST_FILE, TEST_NPC)
    assert upload_success, f"Failed to upload {TEST_FILE}"
    documents = list_documents()
    document_id = documents[0]["document_id"]
    retrieved_doc = get_document_by_id(document_id)
    assert retrieved_doc is not None, f"Failed to retrieve document with ID {document_id}"
    assert retrieved_doc["documentName"] == os.path.basename(TEST_FILE), "Document name mismatch"
    assert int(retrieved_doc["NPC"]) == TEST_NPC, "NPC mismatch"

def test_upload_and_get_by_name():
    """Test uploading a document and then retrieving it by name."""
    # No need to check config here as the fixture will skip the test if needed
    upload_success = upload_document(TEST_FILE, TEST_NPC)
    assert upload_success, f"Failed to upload {TEST_FILE}"
    documents = get_document_by_name(os.path.basename(TEST_FILE))
    assert documents, f"No documents found with name {os.path.basename(TEST_FILE)}"
    assert len(documents) >= 1, "Should retrieve at least one document by name"
    assert int(documents[0]["NPC"]) == TEST_NPC, "NPC mismatch"