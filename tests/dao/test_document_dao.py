"""Tests for DocumentDAO implementations."""

import pytest

from src.models.rag import Document
from src.rag_service.dao.factory import get_document_dao


@pytest.fixture
def document_dao():
    """Get document DAO instance and clean up after tests."""
    dao = get_document_dao()
    yield dao
    # Cleanup: clear mock data if using mock
    if hasattr(dao, "clear") and callable(dao.clear):  # type: ignore[attr-defined]
        dao.clear()  # type: ignore[attr-defined]


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return Document(
        name="test_document.txt",
        agent_id="test-agent-123",
        categories=["General", "Technical"],
    )


def test_create_document(document_dao, sample_document):
    """Test creating a new document."""
    created = document_dao.create(sample_document)

    assert created.id is not None
    assert created.name == sample_document.name
    assert created.agent_id == sample_document.agent_id
    assert created.categories == sample_document.categories
    assert created.created_at is not None
    assert created.updated_at is not None


def test_create_duplicate_document_same_agent(document_dao, sample_document):
    """Test that creating duplicate document for same agent raises error."""
    document_dao.create(sample_document)

    # Try to create another document with same name and agent
    duplicate = Document(
        name=sample_document.name,
        agent_id=sample_document.agent_id,
        categories=["Other"],
    )

    with pytest.raises(ValueError, match="already exists"):
        document_dao.create(duplicate)


def test_create_same_name_different_agent(document_dao, sample_document):
    """Test that same document name can exist for different agents."""
    # Create first document
    doc1 = document_dao.create(sample_document)

    # Create document with same name but different agent
    doc2 = Document(
        name=sample_document.name,
        agent_id="different-agent-456",
        categories=["General"],
    )
    created2 = document_dao.create(doc2)

    assert created2.id is not None
    assert created2.id != doc1.id
    assert created2.agent_id != doc1.agent_id


def test_get_by_id(document_dao, sample_document):
    """Test retrieving document by ID."""
    created = document_dao.create(sample_document)

    retrieved = document_dao.get_by_id(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == created.name
    assert retrieved.agent_id == created.agent_id


def test_get_by_id_not_found(document_dao):
    """Test retrieving non-existent document."""
    result = document_dao.get_by_id("non-existent-id")
    assert result is None


def test_get_by_agent_id(document_dao):
    """Test retrieving all documents for an agent."""
    agent_id = "test-agent-999"

    # Create multiple documents for the agent
    doc1 = Document(name="doc1.txt", agent_id=agent_id, categories=["Cat1"])
    doc2 = Document(name="doc2.txt", agent_id=agent_id, categories=["Cat2"])
    doc3 = Document(name="doc3.txt", agent_id="other-agent", categories=["Cat3"])

    document_dao.create(doc1)
    document_dao.create(doc2)
    document_dao.create(doc3)

    results = document_dao.get_by_agent_id(agent_id)

    assert len(results) == 2
    assert all(doc.agent_id == agent_id for doc in results)


def test_get_by_agent_and_categories(document_dao):
    """Test retrieving documents by agent and categories."""
    agent_id = "test-agent-777"

    # Create documents with different categories
    doc1 = Document(name="doc1.txt", agent_id=agent_id, categories=["General", "Tech"])
    doc2 = Document(name="doc2.txt", agent_id=agent_id, categories=["General"])
    doc3 = Document(name="doc3.txt", agent_id=agent_id, categories=["Other"])
    doc4 = Document(name="doc4.txt", agent_id="other-agent", categories=["General"])

    document_dao.create(doc1)
    document_dao.create(doc2)
    document_dao.create(doc3)
    document_dao.create(doc4)

    # Search for General category
    results = document_dao.get_by_agent_and_categories(agent_id, ["General"])

    assert len(results) == 2  # doc1 and doc2 have "General"
    assert all(doc.agent_id == agent_id for doc in results)
    assert all("General" in doc.categories for doc in results)


def test_get_by_agent_and_categories_multiple(document_dao):
    """Test retrieving documents by agent and multiple categories."""
    agent_id = "test-agent-888"

    doc1 = Document(name="doc1.txt", agent_id=agent_id, categories=["General"])
    doc2 = Document(name="doc2.txt", agent_id=agent_id, categories=["Tech"])
    doc3 = Document(name="doc3.txt", agent_id=agent_id, categories=["Other"])

    document_dao.create(doc1)
    document_dao.create(doc2)
    document_dao.create(doc3)

    # Search for multiple categories
    results = document_dao.get_by_agent_and_categories(agent_id, ["General", "Tech"])

    assert len(results) == 2  # doc1 and doc2
    assert all(doc.agent_id == agent_id for doc in results)


def test_update_document(document_dao, sample_document):
    """Test updating an existing document."""
    created = document_dao.create(sample_document)

    # Update categories
    created.categories = ["Updated", "NewCategory"]
    updated = document_dao.update(created)

    assert updated.categories == ["Updated", "NewCategory"]
    assert updated.updated_at > created.created_at

    # Verify update persisted
    retrieved = document_dao.get_by_id(created.id)
    assert retrieved.categories == ["Updated", "NewCategory"]


def test_update_non_existent_document(document_dao):
    """Test updating a document that doesn't exist."""
    doc = Document(
        id="non-existent-id",
        name="test.txt",
        agent_id="test-agent",
        categories=["Test"],
    )

    with pytest.raises(ValueError, match="not found"):
        document_dao.update(doc)


def test_delete_document(document_dao, sample_document):
    """Test deleting a document."""
    created = document_dao.create(sample_document)

    result = document_dao.delete(created.id)

    assert result is True

    # Verify document is gone
    retrieved = document_dao.get_by_id(created.id)
    assert retrieved is None


def test_delete_non_existent_document(document_dao):
    """Test deleting a document that doesn't exist."""
    result = document_dao.delete("non-existent-id")
    assert result is False


def test_get_by_name_and_agent(document_dao):
    """Test finding document by name and agent."""
    agent_id = "test-agent-555"
    doc_name = "unique_document.txt"

    doc = Document(name=doc_name, agent_id=agent_id, categories=["Test"])
    created = document_dao.create(doc)

    retrieved = document_dao.get_by_name_and_agent(doc_name, agent_id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == doc_name
    assert retrieved.agent_id == agent_id


def test_get_by_name_and_agent_not_found(document_dao):
    """Test finding document by name and agent when it doesn't exist."""
    result = document_dao.get_by_name_and_agent("nonexistent.txt", "test-agent")
    assert result is None


def test_is_reachable(document_dao):
    """Test that DAO is reachable."""
    assert document_dao.is_reachable() is True
