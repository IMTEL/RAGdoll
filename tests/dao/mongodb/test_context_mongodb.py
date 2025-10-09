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

    def test_get_context_by_category(self):
        """Test retrieving contexts by category."""
        # Insert multiple contexts with different categories
        self.repo.insert_context(
            document_id="doc1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for category A",
                category="CategoryA",
                document_name="DocA",
            ),
        )
        self.repo.insert_context(
            document_id="doc2",
            embedding=[0.4, 0.5, 0.6],
            context=Context(
                text="Context for category B",
                category="CategoryB",
                document_name="DocB",
            ),
        )

        # Retrieve contexts by category
        results = self.repo.get_context_by_category("CategoryA")

        assert len(results) == 1
        assert results[0].category == "CategoryA"
        assert results[0].text == "Context for category A"
        assert results[0].document_name == "DocA"
        assert all(isinstance(c, Context) for c in results)

    def test_get_context_by_category_no_results(self):
        """Test retrieving contexts by category when none exist."""
        results = self.repo.get_context_by_category("NonExistentCategory")
        assert results == []
        assert isinstance(results, list)

    def test_get_context_by_category_no_documents(self):
        """Test retrieving contexts when no documents exist in the collection."""
        results = self.repo.get_context_by_category("AnyCategory")
        assert results == []
        assert isinstance(results, list)
