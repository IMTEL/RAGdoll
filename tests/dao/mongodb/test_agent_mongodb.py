"""Comprehensive tests for MongoDBAgentDAO implementation.

These tests verify the MongoDB implementation of the AgentRepository
interface. They require a running MongoDB instance (can be mock or real).
"""

import pytest
from bson import ObjectId

from src.config import Config
from src.models.agents import Agent, Role
from src.rag_service.dao import MongoDBAgentDAO


@pytest.fixture
def mongodb_repo():
    """Create a MongoDB repository instance."""
    return MongoDBAgentDAO()


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    return Agent(
        name="Test MongoDB Agent",
        description="A test agent for MongoDB repository testing",
        prompt="You are a helpful test assistant. User info: {user_information}",
        corpa=["doc1", "doc2", "doc3"],
        roles=[
            Role(
                name="admin",
                description="Administrator with full access",
                subset_of_corpa=[0, 1, 2],
            ),
            Role(
                name="user",
                description="Regular user with limited access",
                subset_of_corpa=[1],
            ),
        ],
        llm_model="gpt-4",
        llm_temperature=0.8,
        llm_max_tokens=2000,
        llm_api_key="test-mongodb-key",
        access_key=["mongodb-key-1", "mongodb-key-2"],
        retrieval_method="semantic",
        embedding_model="text-embedding-ada-002",
        status="active",
        response_format="json",
        last_updated="2025-10-05T12:00:00Z",
    )


@pytest.fixture(autouse=True)
def cleanup_mongodb(mongodb_repo):
    """Clean up MongoDB before and after each test."""
    # Clear the collection before test
    mongodb_repo.collection.delete_many({})
    yield
    # Clear the collection after test
    mongodb_repo.collection.delete_many({})


class TestMongoDBAgentRepositoryConnection:
    """Tests for MongoDB connection and health checks."""

    def test_is_reachable_success(self, mongodb_repo):
        """Test that repository can successfully connect to MongoDB."""
        assert mongodb_repo.is_reachable() is True

    def test_database_configuration(self, mongodb_repo):
        """Verify correct database and collection configuration."""
        config = Config()
        assert mongodb_repo.db.name == config.MONGODB_DATABASE
        assert mongodb_repo.collection.name == config.MONGODB_AGENT_COLLECTION


class TestMongoDBAgentRepositoryCreate:
    """Tests for creating agents in MongoDB."""

    def test_create_agent_success(self, mongodb_repo, sample_agent):
        """Test successful agent creation."""
        created_agent = mongodb_repo.create_agent(sample_agent)

        assert created_agent == sample_agent
        assert created_agent.name == "Test MongoDB Agent"

        # Verify it's actually in the database
        agents_in_db = list(mongodb_repo.collection.find())
        assert len(agents_in_db) == 1
        assert agents_in_db[0]["name"] == "Test MongoDB Agent"

    def test_create_multiple_agents(self, mongodb_repo, sample_agent):
        """Test creating multiple agents."""
        agent1 = sample_agent
        agent2 = Agent(
            name="Second Agent",
            description="Another test agent",
            prompt="Different prompt",
            corpa=["doc4"],
            roles=[],
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.5,
            llm_max_tokens=500,
            llm_api_key="key2",
            access_key=["key"],
            retrieval_method="keyword",
            embedding_model="ada-002",
            status="inactive",
            response_format="text",
            last_updated="2025-10-05T13:00:00Z",
        )

        mongodb_repo.create_agent(agent1)
        mongodb_repo.create_agent(agent2)

        agents_in_db = list(mongodb_repo.collection.find())
        assert len(agents_in_db) == 2
        names = [agent["name"] for agent in agents_in_db]
        assert "Test MongoDB Agent" in names
        assert "Second Agent" in names


class TestMongoDBAgentRepositoryRetrieve:
    """Tests for retrieving agents from MongoDB."""

    def test_get_agents_empty(self, mongodb_repo):
        """Test getting agents from empty database."""
        agents = mongodb_repo.get_agents()
        assert agents == []

    def test_get_agents_single(self, mongodb_repo, sample_agent):
        """Test getting a single agent."""
        mongodb_repo.create_agent(sample_agent)

        agents = mongodb_repo.get_agents()
        assert len(agents) == 1
        assert agents[0].name == "Test MongoDB Agent"
        assert agents[0].llm_model == "gpt-4"
        assert len(agents[0].roles) == 2

    def test_get_agents_multiple(self, mongodb_repo, sample_agent):
        """Test getting multiple agents."""
        agent1 = sample_agent
        agent2 = Agent(
            name="Agent 2",
            description="Second agent",
            prompt="Prompt 2",
            corpa=[],
            roles=[],
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            llm_api_key="key2",
            access_key=[],
            retrieval_method="semantic",
            embedding_model="ada-002",
            status="active",
            response_format="text",
            last_updated="2025-10-05T14:00:00Z",
        )

        mongodb_repo.create_agent(agent1)
        mongodb_repo.create_agent(agent2)

        agents = mongodb_repo.get_agents()
        assert len(agents) == 2
        names = [agent.name for agent in agents]
        assert "Test MongoDB Agent" in names
        assert "Agent 2" in names

    def test_get_agent_by_id_success(self, mongodb_repo, sample_agent):
        """Test successfully retrieving an agent by ID."""
        # Create agent and get its ID
        result = mongodb_repo.collection.insert_one(sample_agent.model_dump())
        agent_id = str(result.inserted_id)

        # Retrieve by ID
        retrieved_agent = mongodb_repo.get_agent_by_id(agent_id)

        assert retrieved_agent is not None
        assert retrieved_agent.name == "Test MongoDB Agent"
        assert retrieved_agent.llm_model == "gpt-4"
        assert retrieved_agent.llm_temperature == 0.8

    def test_get_agent_by_id_not_found(self, mongodb_repo):
        """Test retrieving non-existent agent returns None."""
        fake_id = str(ObjectId())
        agent = mongodb_repo.get_agent_by_id(fake_id)
        assert agent is None

    def test_get_agent_by_id_invalid_format(self, mongodb_repo):
        """Test retrieving agent with invalid ID format returns None."""
        invalid_id = "not-a-valid-objectid"
        agent = mongodb_repo.get_agent_by_id(invalid_id)
        assert agent is None

    def test_agent_roles_preserved(self, mongodb_repo, sample_agent):
        """Test that agent roles are correctly preserved."""
        result = mongodb_repo.collection.insert_one(sample_agent.model_dump())
        agent_id = str(result.inserted_id)

        retrieved_agent = mongodb_repo.get_agent_by_id(agent_id)

        assert retrieved_agent is not None
        assert len(retrieved_agent.roles) == 2
        assert retrieved_agent.roles[0].name == "admin"
        assert retrieved_agent.roles[0].subset_of_corpa == [0, 1, 2]
        assert retrieved_agent.roles[1].name == "user"
        assert retrieved_agent.roles[1].subset_of_corpa == [1]

    def test_agent_corpa_preserved(self, mongodb_repo, sample_agent):
        """Test that agent corpus list is correctly preserved."""
        result = mongodb_repo.collection.insert_one(sample_agent.model_dump())
        agent_id = str(result.inserted_id)

        retrieved_agent = mongodb_repo.get_agent_by_id(agent_id)

        assert retrieved_agent is not None
        assert retrieved_agent.corpa == ["doc1", "doc2", "doc3"]


class TestMongoDBAgentRepositoryEdgeCases:
    """Tests for edge cases and error handling."""

    def test_agent_with_empty_roles(self, mongodb_repo):
        """Test creating and retrieving agent with no roles."""
        agent = Agent(
            name="No Roles Agent",
            description="Agent without roles",
            prompt="Simple prompt",
            corpa=["doc1"],
            roles=[],
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            llm_api_key="key",
            access_key=[],
            retrieval_method="semantic",
            embedding_model="ada-002",
            status="active",
            response_format="text",
            last_updated="2025-10-05T15:00:00Z",
        )

        mongodb_repo.create_agent(agent)
        agents = mongodb_repo.get_agents()

        assert len(agents) == 1
        assert agents[0].roles == []

    def test_agent_with_empty_corpa(self, mongodb_repo):
        """Test creating and retrieving agent with no corpus."""
        agent = Agent(
            name="No Corpus Agent",
            description="Agent without corpus",
            prompt="Simple prompt",
            corpa=[],
            roles=[],
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            llm_api_key="key",
            access_key=[],
            retrieval_method="semantic",
            embedding_model="ada-002",
            status="active",
            response_format="text",
            last_updated="2025-10-05T15:00:00Z",
        )

        mongodb_repo.create_agent(agent)
        agents = mongodb_repo.get_agents()

        assert len(agents) == 1
        assert agents[0].corpa == []
