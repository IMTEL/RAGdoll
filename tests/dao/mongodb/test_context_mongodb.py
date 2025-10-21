# FIXME: These tests are outdated and need to be revised.

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

        # Verify it was inserted
        count = self.repo.collection.count_documents({})
        assert count == 1

    def test_get_context_for_agent(self):
        """Test retrieving contexts for a specific agent."""
        # Insert multiple contexts for different agents
        self.repo.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )
        self.repo.insert_context(
            document_id="doc2",
            agent_id="agent-2",
            embedding=[0.4, 0.5, 0.6],
            context=Context(
                text="Context for agent 2",
                document_name="DocB",
                document_id="doc2",
            ),
        )

        # Retrieve contexts for agent-1 with its document
        results = self.repo.get_context_for_agent(
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
        """Test retrieving contexts for agent when no accessible documents exist."""
        # Insert context for agent-1
        self.repo.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )

        # Try to get contexts with a different agent_id
        results = self.repo.get_context_for_agent(
            agent_id="agent-999",
            embedding=[0.1, 0.2, 0.3],
            documents=["doc1"],
            top_k=5,
        )

        assert results == []
        assert isinstance(results, list)

    def test_get_context_for_agent_no_documents(self):
        """Test retrieving contexts when no documents exist in the collection."""
        results = self.repo.get_context_for_agent(
            agent_id="agent-1",
            embedding=[0.1, 0.2, 0.3],
            documents=[],
            top_k=5,
        )

        assert results == []
        assert isinstance(results, list)
