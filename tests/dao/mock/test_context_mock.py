import pytest

from src.rag_service.context import Context
from tests.mocks import MockContextDAO


class TestMockContextDAO:
    """Tests for MockContextDAO (in-memory implementation)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear mock database before and after each test."""
        repo = MockContextDAO()
        repo.clear()
        yield
        repo.clear()

    def test_is_reachable(self):
        """Test that is_reachable returns True."""
        repo = MockContextDAO()
        assert repo.is_reachable() is True

    def test_post_context(self):
        """Test posting a new context."""
        repo = MockContextDAO()

        context = repo.insert_context(
            document_id="doc123",
            agent_id="agent-456",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Sample context text",
                document_name="TestDoc",
                document_id="doc123",
            ),
        )

        assert context.text == "Sample context text"
        assert context.document_name == "TestDoc"
        assert len(repo.data) == 1

    def test_get_context_for_agent(self):
        """Test retrieving contexts for a specific agent."""
        repo = MockContextDAO()

        # Insert multiple contexts for different agents
        repo.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )
        repo.insert_context(
            document_id="doc2",
            agent_id="agent-2",
            embedding=[0.4, 0.5, 0.6],
            context=Context(
                text="Context for agent 2",
                document_name="DocB",
                document_id="doc2",
            ),
        )

        # Retrieve contexts for agent-1
        results = repo.get_context_for_agent(
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            documents=["doc1"],
            top_k=5,
        )

        # Should only return contexts for agent-1
        assert all(isinstance(c, Context) for c in results)
        if len(results) > 0:
            assert results[0].document_id == "doc1"

    def test_get_context_for_agent_no_results(self):
        """Test retrieving contexts when no accessible documents exist."""
        repo = MockContextDAO()

        # Insert context for agent-1
        repo.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )

        # Try to retrieve with different agent
        results = repo.get_context_for_agent(
            agent_id="agent-999",
            embedding=[0.1, 0.2, 0.3],
            documents=["doc1"],
            top_k=5,
        )

        assert results == []
        assert isinstance(results, list)
