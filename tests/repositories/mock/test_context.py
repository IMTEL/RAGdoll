import pytest

from src.rag_service.context import Context
from tests.mocks import MockContextRepository


class TestMockContextRepository:
    """Tests for MockContextRepository (in-memory implementation)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear mock database before and after each test."""
        repo = MockContextRepository()
        repo.clear()
        yield
        repo.clear()

    def test_is_reachable(self):
        """Test that is_reachable returns True."""
        repo = MockContextRepository()
        assert repo.is_reachable() is True

    def test_post_context(self):
        """Test posting a new context."""
        repo = MockContextRepository()

        context = repo.insert_context(
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
        assert len(repo.data) == 1

    def test_get_context_by_category(self):
        """Test retrieving contexts by category."""
        repo = MockContextRepository()

        # Insert multiple contexts with different categories
        repo.insert_context(
            document_id="doc1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for category A",
                category="CategoryA",
                document_name="DocA",
            ),
        )
        repo.insert_context(
            document_id="doc2",
            embedding=[0.4, 0.5, 0.6],
            context=Context(
                text="Context for category B",
                category="CategoryB",
                document_name="DocB",
            ),
        )

        # Retrieve contexts by category
        results = repo.get_context_by_category("CategoryA")

        assert len(results) == 1
        assert results[0].category == "CategoryA"
        assert results[0].text == "Context for category A"
        assert results[0].document_name == "DocA"
        assert all(isinstance(c, Context) for c in results)

    def test_get_context_by_category_no_results(self):
        """Test retrieving contexts by category when none exist."""
        repo = MockContextRepository()

        results = repo.get_context_by_category("NonExistentCategory")

        assert results == []
        assert isinstance(results, list)
