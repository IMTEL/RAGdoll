import pytest
import time
from uuid import uuid4
from src.rag_service.dao import get_database, MockDatabase
from src.rag_service.context import Context

# --- Helper Classes and Fixtures ---

class FakeCollection:
    """
    A fake collection to simulate MongoDB's aggregate() method.
    It simply returns an iterator over the provided data.
    """
    def __init__(self, data):
        self.data = data

    def aggregate(self, pipeline):
        # In a real scenario, you'd parse the pipeline;
        # here we just return all stored documents.
        return iter(self.data)

@pytest.fixture
def mock_db():
    """
    Fixture that returns a MockDatabase instance.
    If get_database() returns a non-mock instance, skip the tests.
    """
    db = get_database()
    if not isinstance(db, MockDatabase):
        pytest.skip("Skipping tests because get_database() did not return a MockDatabase instance")
    # Attach a FakeCollection based on its in-memory data
    data = db.collection.data
    db.collection = FakeCollection(data)
    return db


def test_post_context_invalid_params(mock_db):
    """
    Test that post_context raises a ValueError when required parameters are missing or invalid.
    """
    # Empty text should trigger a ValueError
    with pytest.raises(ValueError) as exc_info:
        mock_db.post_context(
            text="",
            NPC=1,
            embedding=[0.1, 0.1],
            document_id="some_id",
            document_name="TestDoc"
        )
    assert "text cannot be None" in str(exc_info.value)

def test_is_reachable(mock_db):
    """
    Test that is_reachable returns True for the mock database.
    """
    assert mock_db.is_reachable() is True


