import pytest

from src.rag_service.context import Context
from src.rag_service.dao.context.mongodb_context_dao import (
    MongoDBContextDAO,
)


class TestMongoDBContextDAO:
    """Tests for MongoDBContextDAO."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for MongoDB tests."""
        self.repo = MongoDBContextDAO()

        if not self.repo.is_reachable():
            pytest.skip("MongoDB is not reachable. Skipping tests.")

        # Clear the collection before and after each test
        self.repo.collection.delete_many({})
        yield
        self.repo.collection.delete_many({})

    def test_is_reachable(self):
        """Test that is_reachable returns True."""
        assert self.repo.is_reachable() is True

    def test_post_context(self):
        """Test posting a new context."""
        context = self.repo.insert_context(
            document_id="doc123",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Sample context text",
                category="General Information",
                document_name="TestDoc",
            ),
        )

        assert context.text == "Sample context text"
        assert context.category == "General Information"

        # Verify it was inserted
        count = self.repo.collection.count_documents({})
        assert count == 1

    # TODO: Add tests for retrieving context based on roles of an agent
