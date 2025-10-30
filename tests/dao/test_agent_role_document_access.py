"""Tests for Agent Role Document Access system.

These tests verify the new document access control system where:
- Agents no longer have a corpus field
- Roles have document_access lists containing document IDs
- Document access is controlled at the role level
"""

import pytest

from src.models.agent import Agent, Role
from src.models.rag import Document
from src.rag_service.dao.factory import get_agent_dao, get_document_dao


@pytest.fixture
def agent_dao():
    """Get agent DAO instance and clean up."""
    dao = get_agent_dao()

    # Clean up before test
    if hasattr(dao, "collection"):
        dao.collection.delete_many({})
    elif hasattr(dao, "clear"):
        dao.clear()

    yield dao

    # Clean up after test
    if hasattr(dao, "collection"):
        dao.collection.delete_many({})
    elif hasattr(dao, "clear"):
        dao.clear()


@pytest.fixture
def document_dao():
    """Get document DAO instance and clean up."""
    dao = get_document_dao()

    # Clean up before test
    if hasattr(dao, "collection"):
        dao.collection.delete_many({})
    elif hasattr(dao, "clear"):
        dao.clear()

    yield dao

    # Clean up after test
    if hasattr(dao, "collection"):
        dao.collection.delete_many({})
    elif hasattr(dao, "clear"):
        dao.clear()


class TestRoleDocumentAccess:
    """Tests for role-based document access control."""

    def test_role_with_document_access(self, agent_dao):
        """Test creating a role with document access list."""
        role = Role(
            name="viewer",
            description="View specific documents",
            document_access=["doc-id-1", "doc-id-2", "doc-id-3"],
        )

        agent = Agent(
            name="Test Agent",
            description="Test agent with role",
            prompt="Test prompt",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        assert len(created.roles) == 1
        assert created.roles[0].document_access == ["doc-id-1", "doc-id-2", "doc-id-3"]

    def test_role_with_empty_document_access(self, agent_dao):
        """Test creating a role with no document access."""
        role = Role(
            name="restricted", description="No document access", document_access=[]
        )

        agent = Agent(
            name="Restricted Agent",
            description="Agent with restricted role",
            prompt="Test prompt",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        assert len(created.roles) == 1
        assert created.roles[0].document_access == []

    def test_multiple_roles_different_access(self, agent_dao):
        """Test agent with multiple roles having different document access."""
        admin_role = Role(
            name="admin",
            description="Full access",
            document_access=["doc-1", "doc-2", "doc-3", "doc-4"],
        )

        user_role = Role(
            name="user",
            description="Limited access",
            document_access=["doc-2", "doc-3"],
        )

        guest_role = Role(
            name="guest", description="Minimal access", document_access=["doc-3"]
        )

        agent = Agent(
            name="Multi-Role Agent",
            description="Agent with multiple roles",
            prompt="Test prompt",
            roles=[admin_role, user_role, guest_role],
            llm_model="gpt-4",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        assert len(created.roles) == 3
        assert created.roles[0].document_access == ["doc-1", "doc-2", "doc-3", "doc-4"]
        assert created.roles[1].document_access == ["doc-2", "doc-3"]
        assert created.roles[2].document_access == ["doc-3"]

    def test_agent_get_role_by_name(self, agent_dao):
        """Test retrieving a specific role from an agent."""
        admin_role = Role(
            name="admin",
            description="Administrator",
            document_access=["doc-1", "doc-2"],
        )

        user_role = Role(
            name="user", description="Regular user", document_access=["doc-2"]
        )

        agent = Agent(
            name="Test Agent",
            description="Test",
            prompt="Test",
            roles=[admin_role, user_role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        # Test the get_role_by_name method
        admin = agent.get_role_by_name("admin")
        user = agent.get_role_by_name("user")
        nonexistent = agent.get_role_by_name("nonexistent")

        assert admin is not None
        assert admin.name == "admin"
        assert admin.document_access == ["doc-1", "doc-2"]

        assert user is not None
        assert user.name == "user"
        assert user.document_access == ["doc-2"]

        assert nonexistent is None

    def test_role_document_access_persistence(self, agent_dao):
        """Test that document access lists persist through save/retrieve cycle."""
        role = Role(
            name="tester",
            description="Test role",
            document_access=["alpha", "beta", "gamma", "delta"],
        )

        agent = Agent(
            name="Persistence Test Agent",
            description="Testing persistence",
            prompt="Test",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        agent_id = created.id

        # Retrieve the agent
        retrieved = agent_dao.get_agent_by_id(agent_id)

        assert retrieved is not None
        assert len(retrieved.roles) == 1
        assert retrieved.roles[0].name == "tester"
        assert retrieved.roles[0].document_access == ["alpha", "beta", "gamma", "delta"]


class TestAgentDocumentRelationship:
    """Tests for the relationship between agents and documents."""

    def test_documents_linked_to_agent(self, agent_dao, document_dao):
        """Test that documents are properly linked to their agent."""
        # Create an agent
        agent = Agent(
            name="Document Owner",
            description="Agent owning documents",
            prompt="Test",
            roles=[],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )
        created_agent = agent_dao.add_agent(agent)
        agent_id = created_agent.id

        # Create documents for this agent
        doc1 = Document(name="doc1.txt", agent_id=agent_id)
        doc2 = Document(name="doc2.txt", agent_id=agent_id)
        doc3 = Document(name="doc3.txt", agent_id=agent_id)

        created_doc1 = document_dao.create(doc1)
        created_doc2 = document_dao.create(doc2)
        created_doc3 = document_dao.create(doc3)

        # Retrieve all documents for this agent
        agent_docs = document_dao.get_by_agent_id(agent_id)

        assert len(agent_docs) == 3
        doc_ids = {doc.id for doc in agent_docs}
        assert created_doc1.id in doc_ids
        assert created_doc2.id in doc_ids
        assert created_doc3.id in doc_ids

    def test_role_references_actual_documents(self, agent_dao, document_dao):
        """Test workflow where role's document_access references real document IDs."""
        # Create an agent first (without documents in roles yet)
        agent = Agent(
            name="Test Agent",
            description="Test",
            prompt="Test",
            roles=[
                Role(
                    name="user",
                    description="User role",
                    document_access=[],  # Empty initially
                )
            ],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )
        created_agent = agent_dao.add_agent(agent)
        agent_id = created_agent.id

        # Create documents for this agent
        doc1 = Document(name="manual.pdf", agent_id=agent_id)
        doc2 = Document(name="guide.pdf", agent_id=agent_id)

        created_doc1 = document_dao.create(doc1)
        created_doc2 = document_dao.create(doc2)

        # Update agent's role to reference these documents
        retrieved_agent = agent_dao.get_agent_by_id(agent_id)
        retrieved_agent.roles[0].document_access = [created_doc1.id, created_doc2.id]

        # Save the updated agent
        agent_dao.add_agent(retrieved_agent)

        # Verify the update persisted
        final_agent = agent_dao.get_agent_by_id(agent_id)
        assert len(final_agent.roles[0].document_access) == 2
        assert created_doc1.id in final_agent.roles[0].document_access
        assert created_doc2.id in final_agent.roles[0].document_access

    def test_document_access_with_nonexistent_documents(self, agent_dao):
        """Test that roles can reference document IDs that don't exist yet.

        This is valid because documents might be added later, or the role
        might be set up before documents are uploaded.
        """
        role = Role(
            name="preparer",
            description="Role prepared for future documents",
            document_access=["future-doc-1", "future-doc-2", "future-doc-3"],
        )

        agent = Agent(
            name="Forward-looking Agent",
            description="Agent with roles referencing future documents",
            prompt="Test",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)

        # This should work fine - no validation that documents exist
        assert created.roles[0].document_access == [
            "future-doc-1",
            "future-doc-2",
            "future-doc-3",
        ]


class TestDocumentAccessEdgeCases:
    """Tests for edge cases in document access control."""

    def test_role_with_duplicate_document_ids(self, agent_dao):
        """Test role with duplicate document IDs in access list."""
        role = Role(
            name="duplicates",
            description="Role with duplicate IDs",
            document_access=["doc-1", "doc-2", "doc-1", "doc-3", "doc-2"],
        )

        agent = Agent(
            name="Test Agent",
            description="Test",
            prompt="Test",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)

        # Should store exactly as given (duplicates allowed)
        # The application layer can deduplicate if needed
        assert created.roles[0].document_access == [
            "doc-1",
            "doc-2",
            "doc-1",
            "doc-3",
            "doc-2",
        ]

    def test_role_document_access_large_list(self, agent_dao):
        """Test role with many document IDs."""
        # Create a large document access list
        large_list = [f"doc-{i}" for i in range(1000)]

        role = Role(
            name="power_user",
            description="Access to many documents",
            document_access=large_list,
        )

        agent = Agent(
            name="Power User Agent",
            description="Agent with access to many documents",
            prompt="Test",
            roles=[role],
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        assert len(created.roles[0].document_access) == 1000
        assert created.roles[0].document_access[0] == "doc-0"
        assert created.roles[0].document_access[-1] == "doc-999"

    def test_agent_without_roles(self, agent_dao):
        """Test agent with no roles defined."""
        agent = Agent(
            name="Roleless Agent",
            description="Agent without any roles",
            prompt="Test",
            roles=[],  # Empty roles list
            llm_model="gpt-3.5-turbo",
            llm_api_key="test-key",
            access_key=[],
            embedding_model="ada-002",
            last_updated="2025-10-21T10:00:00Z",
            embedding_api_key="test-embedding-key",
        )

        created = agent_dao.add_agent(agent)
        assert created.roles == []

        # get_role_by_name should return None for any role name
        assert created.get_role_by_name("any_role") is None
