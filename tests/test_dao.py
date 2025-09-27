from unittest.mock import MagicMock, patch

import pytest

from src.rag_service.dao import MockDatabase, get_database


class FakeCollection:
    """A fake collection to simulate MongoDB's aggregate() and insert_one() methods."""

    def __init__(self, data=None):
        self.data = data if data is not None else []

    def aggregate(self, pipeline):
        # In a real scenario, you'd parse the pipeline;
        # here we just return all stored documents.
        return iter(self.data)

    def insert_one(self, document):
        # Simulate MongoDB insert_one operation
        self.data.append(document)
        return MagicMock()


@pytest.fixture
def mock_db():
    """Fixture that returns a MockDatabase instance.

    If get_database() returns a non-mock instance, skip the tests.
    """
    db = get_database()
    if not isinstance(db, MockDatabase):
        pytest.skip(
            "Skipping tests because get_database() did not return a MockDatabase instance"
        )

    # Create a new FakeCollection with empty data
    fake_collection = FakeCollection([])
    # Save the original collection to restore later
    original_collection = db.collection
    # Set the fake collection
    db.collection = fake_collection

    yield db

    # Restore the original collection after the test
    db.collection = original_collection


def test_post_context_invalid_params(mock_db):
    """Test that post_context raises a ValueError when required parameters are missing or invalid."""
    # Empty text should trigger a ValueError
    with pytest.raises(ValueError) as exc_info:
        mock_db.post_context(
            text="",
            category="General Information",
            embedding=[0.1, 0.1],
            document_id="some_id",
            document_name="TestDoc",
        )
    assert "text cannot be None" in str(exc_info.value)


def test_is_reachable(mock_db):
    """Test that is_reachable returns True for the mock database."""
    assert mock_db.is_reachable() is True


@patch(
    "src.rag_service.dao.similarity_search", return_value=0.8
)  # Mock the similarity_search to return 0.8
def test_get_context_by_category(similarity_search_mock, mock_db):
    """Test that get_context_by_category returns contexts with the specified category."""
    # Add test data directly to the fake collection
    test_category = "FishFeeding"
    document = {
        "text": "Test text for FishFeeding scene",
        "category": "FishFeeding",
        "embedding": [0.1, 0.1, 0.3],
        "documentId": "test_id_1",
        "documentName": "TestDoc1",
    }
    if mock_db.__class__ is not MockDatabase:
        pytest.skip("Not using MockDatabase; skipping DAO unit tests")
    mock_db.data.append(document)

    # Try to retrieve by category - this should work now with our fake collection
    contexts = mock_db.get_context_by_category(test_category)

    # Assertions - check that we got the expected data back
    assert len(contexts) > 0
    assert contexts[0].category == test_category
    assert contexts[0].text == "Test text for FishFeeding scene"
