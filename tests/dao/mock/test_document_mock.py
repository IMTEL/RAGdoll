"""Tests for MockDocumentDAO implementation."""

import pytest

from src.models.rag import Document
from tests.mocks import MockDocumentDAO


class TestMockDocumentDAO:
    """Tests for MockDocumentDAO (in-memory implementation)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for mock tests."""
        self.repo = MockDocumentDAO()
        self.repo.clear()
        yield
        self.repo.clear()

    def test_is_reachable(self):
        """Test that DAO is always reachable."""
        assert self.repo.is_reachable() is True

    def test_create_document(self):
        """Test creating a new document."""
        doc = Document(
            name="test_doc.txt",
            agent_id="agent-123",
        )

        created = self.repo.create(doc)

        assert created.id is not None
        assert created.name == "test_doc.txt"
        assert created.agent_id == "agent-123"
        assert created.created_at is not None
        assert created.updated_at is not None

    def test_create_duplicate_document(self):
        """Test that creating duplicate document for same agent raises error."""
        doc1 = Document(
            name="duplicate.txt",
            agent_id="agent-123",
        )
        self.repo.create(doc1)

        # Try to create duplicate
        doc2 = Document(
            name="duplicate.txt",
            agent_id="agent-123",
        )

        with pytest.raises(ValueError, match="already exists"):
            self.repo.create(doc2)

    def test_create_same_name_different_agent(self):
        """Test that same document name can exist for different agents."""
        doc1 = Document(
            name="shared_name.txt",
            agent_id="agent-1",
        )
        created1 = self.repo.create(doc1)

        doc2 = Document(
            name="shared_name.txt",
            agent_id="agent-2",
        )
        created2 = self.repo.create(doc2)

        assert created1.id != created2.id
        assert created1.agent_id == "agent-1"
        assert created2.agent_id == "agent-2"

    def test_get_by_id(self):
        """Test retrieving document by ID."""
        doc = Document(name="doc.txt", agent_id="agent-123")
        created = self.repo.create(doc)

        retrieved = self.repo.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == created.name
        assert retrieved.agent_id == created.agent_id

    def test_get_by_id_not_found(self):
        """Test retrieving non-existent document."""
        result = self.repo.get_by_id("non-existent-id")
        assert result is None

    def test_get_by_agent_id(self):
        """Test retrieving all documents for an agent."""
        # Create documents for different agents
        doc1 = Document(name="doc1.txt", agent_id="agent-1")
        doc2 = Document(name="doc2.txt", agent_id="agent-1")
        doc3 = Document(name="doc3.txt", agent_id="agent-2")

        self.repo.create(doc1)
        self.repo.create(doc2)
        self.repo.create(doc3)

        # Get documents for agent-1
        results = self.repo.get_by_agent_id("agent-1")

        assert len(results) == 2
        assert all(doc.agent_id == "agent-1" for doc in results)
        doc_names = {doc.name for doc in results}
        assert doc_names == {"doc1.txt", "doc2.txt"}

    def test_get_by_agent_id_empty(self):
        """Test retrieving documents for agent with no documents."""
        results = self.repo.get_by_agent_id("non-existent-agent")
        assert results == []

    def test_get_by_name_and_agent(self):
        """Test retrieving document by name and agent."""
        doc = Document(name="specific.txt", agent_id="agent-123")
        created = self.repo.create(doc)

        retrieved = self.repo.get_by_name_and_agent("specific.txt", "agent-123")

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "specific.txt"

    def test_get_by_name_and_agent_not_found(self):
        """Test retrieving non-existent document by name and agent."""
        result = self.repo.get_by_name_and_agent("missing.txt", "agent-123")
        assert result is None

    def test_update_document(self):
        """Test updating an existing document."""
        import time

        doc = Document(name="doc.txt", agent_id="agent-123")
        created = self.repo.create(doc)

        # Small delay to ensure timestamp difference
        time.sleep(0.001)

        # Update name
        created.name = "updated_doc.txt"
        updated = self.repo.update(created)

        assert updated.id == created.id
        assert updated.name == "updated_doc.txt"
        assert (
            updated.updated_at >= created.created_at
        )  # Use >= to handle fast execution

        # Verify update persisted
        retrieved = self.repo.get_by_id(created.id)
        assert retrieved.name == "updated_doc.txt"

    def test_update_nonexistent_document(self):
        """Test updating a document that doesn't exist."""
        doc = Document(
            id="non-existent-id",
            name="doc.txt",
            agent_id="agent-123",
        )

        with pytest.raises(ValueError, match="not found"):
            self.repo.update(doc)

    def test_delete_document(self):
        """Test deleting a document."""
        doc = Document(name="to_delete.txt", agent_id="agent-123")
        created = self.repo.create(doc)

        # Delete the document
        self.repo.delete(created.id)

        # Verify it's gone
        retrieved = self.repo.get_by_id(created.id)
        assert retrieved is None

    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist."""
        # Should not raise an error, just do nothing
        self.repo.delete("non-existent-id")

    def test_singleton_pattern(self):
        """Test that MockDocumentDAO follows singleton pattern."""
        dao1 = MockDocumentDAO()
        dao2 = MockDocumentDAO()

        assert dao1 is dao2

        # Test that data persists across instances
        doc = Document(name="test.txt", agent_id="agent-123")
        dao1.create(doc)

        docs = dao2.get_by_agent_id("agent-123")
        assert len(docs) == 1
        assert docs[0].name == "test.txt"

    def test_clear(self):
        """Test clearing all documents."""
        doc1 = Document(name="doc1.txt", agent_id="agent-1")
        doc2 = Document(name="doc2.txt", agent_id="agent-2")

        self.repo.create(doc1)
        self.repo.create(doc2)

        assert len(self.repo.get_by_agent_id("agent-1")) == 1
        assert len(self.repo.get_by_agent_id("agent-2")) == 1

        self.repo.clear()

        assert self.repo.get_by_agent_id("agent-1") == []
        assert self.repo.get_by_agent_id("agent-2") == []
