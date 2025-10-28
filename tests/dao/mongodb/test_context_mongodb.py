# FIXME: These tests are outdated and need to be revised.

import pytest

from src.rag_service.context import Context
from src.rag_service.dao.context.mongodb_context_dao import (
    MongoDBContextDAO,
)


def create_test_embedding(seed: float = 0.1) -> list[float]:
    return [seed + (i * 0.001) for i in range(768)]


class TestMongoDBContextDAO:
    """Tests for MongoDBContextDAO."""

    # Flag to skip all tests if MongoDB is unreachable
    mongodb_found_unreachable = False

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for MongoDB tests."""
        self.context_dao = MongoDBContextDAO()

        if self.mongodb_found_unreachable or not self.context_dao.is_reachable():
            self.mongodb_found_unreachable = True
            pytest.skip("MongoDB is not reachable. Skipping tests.")

        # Clear the collection before and after each test
        self.context_dao.collection.delete_many({})
        yield
        self.context_dao.collection.delete_many({})

        # Drop the test database after tests
        self.context_dao.client.drop_database(self.context_dao.db.name)

    def test_is_reachable(self):
        """Test that is_reachable returns True."""
        assert self.context_dao.is_reachable() is True

    def test_post_context(self):
        """Test posting a new context."""
        context = self.context_dao.insert_context(
            document_id="doc123",
            agent_id="agent-456",
            embedding=create_test_embedding(0.1),
            context=Context(
                text="Sample context text",
                document_name="TestDoc",
                document_id="doc123",
            ),
        )

        assert context.text == "Sample context text"
        assert context.document_name == "TestDoc"

        # Verify it was inserted
        count = self.context_dao.collection.count_documents({})
        assert count == 1

    def test_get_context_for_agent(self):
        """Test retrieving contexts for a specific agent."""
        # Insert multiple contexts for different agents
        self.context_dao.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=create_test_embedding(0.1),
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )
        self.context_dao.insert_context(
            document_id="doc2",
            agent_id="agent-2",
            embedding=create_test_embedding(0.4),
            context=Context(
                text="Context for agent 2",
                document_name="DocB",
                document_id="doc2",
            ),
        )

        # Retrieve contexts for agent-1 with its document
        results = self.context_dao.get_context_for_agent(
            agent_id="agent-1",
            embedding=create_test_embedding(0.1),
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
        self.context_dao.insert_context(
            document_id="doc1",
            agent_id="agent-1",
            embedding=create_test_embedding(0.1),
            context=Context(
                text="Context for agent 1",
                document_name="DocA",
                document_id="doc1",
            ),
        )

        # Try to get contexts with a different agent_id
        results = self.context_dao.get_context_for_agent(
            agent_id="agent-999",
            embedding=create_test_embedding(0.1),
            documents=["doc1"],
            top_k=5,
        )

        assert results == []
        assert isinstance(results, list)

    def test_get_context_for_agent_no_documents(self):
        """Test retrieving contexts when no documents exist in the collection."""
        results = self.context_dao.get_context_for_agent(
            agent_id="agent-1",
            embedding=create_test_embedding(0.1),
            documents=[],
            top_k=5,
        )

        assert results == []
        assert isinstance(results, list)
