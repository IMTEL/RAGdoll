"""Comprehensive tests for AgentRepository implementations.

Tests both MongoDBAgentRepository and MockAgentRepository to ensure
they properly implement the repository pattern.
"""

import pytest

from src.models.agent import Agent
from src.models.role import Role
from src.rag_service.repositories import get_agent_repository
from tests.mocks import MockAgentRepository


@pytest.fixture(autouse=True)
def clear_mock_agent_repository():
    """Clear mock repository before and after each test."""
    repo = get_agent_repository()
    if isinstance(repo, MockAgentRepository):
        repo.clear()
    yield
    if isinstance(repo, MockAgentRepository):
        repo.clear()


@pytest.fixture
def sample_agent():
    """Create a sample agent for testing."""
    return Agent(
        name="Test Bot",
        description="A test bot for unit testing",
        prompt="You are a helpful assistant",
        corpa=["document1", "document2"],
        roles=[
            Role(
                name="admin", description="Administrator role", subset_of_corpa=[0, 1]
            ),
            Role(name="user", description="User role", subset_of_corpa=[1]),
        ],
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.7,
        llm_max_tokens=1000,
        llm_api_key="test-api-key",
        access_key=["key1", "key2"],
        retrieval_method="semantic",
        embedding_model="text-embedding-ada-002",
        status="active",
        response_format="json",
        last_updated="2025-10-03T10:00:00Z",
    )


class TestMockAgentRepository:
    """Tests for MockAgentRepository (in-memory implementation)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Clear mock database before and after each test."""
        repo = MockAgentRepository()
        repo.clear()
        yield
        repo.clear()

    def test_is_reachable(self):
        """Test that repository is always reachable."""
        repo = MockAgentRepository()
        assert repo.is_reachable() is True

    def test_create_agent(self, sample_agent):
        """Test creating an agent."""
        repo = MockAgentRepository()

        result = repo.create_agent(sample_agent)

        assert isinstance(result, Agent)
        assert result.name == "Test Bot"
        assert result.description == "A test bot for unit testing"
        assert len(repo.agents) == 1

    def test_create_multiple_agents(self, sample_agent):
        """Test creating multiple agents."""
        repo = MockAgentRepository()

        agent1 = sample_agent
        agent2 = Agent(**sample_agent.model_dump())
        agent2.name = "Second Bot"
        agent2.description = "Another test bot"

        repo.create_agent(agent1)
        repo.create_agent(agent2)

        assert len(repo.agents) == 2

    def test_get_agents_empty(self):
        """Test getting agents when repository is empty."""
        repo = MockAgentRepository()

        agents = repo.get_agents()

        assert agents == []
        assert isinstance(agents, list)

    def test_get_agents(self, sample_agent):
        """Test retrieving all agents."""
        repo = MockAgentRepository()

        # Create multiple agents
        agent1 = sample_agent
        agent2 = Agent(**sample_agent.model_dump())
        agent2.name = "Second Bot"

        repo.create_agent(agent1)
        repo.create_agent(agent2)

        # Retrieve all
        agents = repo.get_agents()

        assert len(agents) == 2
        assert all(isinstance(a, Agent) for a in agents)
        assert agents[0].name == "Test Bot"
        assert agents[1].name == "Second Bot"

    def test_get_agent_by_id(self, sample_agent):
        """Test retrieving a specific agent by ID."""
        repo = MockAgentRepository()

        # Create agents
        repo.create_agent(sample_agent)
        agent2 = Agent(**sample_agent.model_dump())
        agent2.name = "Second Bot"
        repo.create_agent(agent2)

        # Retrieve by ID (index 0)
        agent = repo.get_agent_by_id("0")

        assert agent is not None
        assert isinstance(agent, Agent)
        assert agent.name == "Test Bot"

    def test_get_agent_by_id_not_found(self):
        """Test retrieving non-existent agent returns None."""
        repo = MockAgentRepository()

        agent = repo.get_agent_by_id("999")

        assert agent is None

    def test_get_agent_by_invalid_id(self, sample_agent):
        """Test retrieving agent with invalid ID format."""
        repo = MockAgentRepository()
        repo.create_agent(sample_agent)

        agent = repo.get_agent_by_id("invalid")

        assert agent is None

    def test_repository_singleton(self, sample_agent):
        """Test that repository uses singleton pattern."""
        repo1 = MockAgentRepository()
        repo2 = MockAgentRepository()

        # They should be the same instance (singleton)
        assert repo1 is repo2

        # Changes in one should reflect in the other
        repo1.create_agent(sample_agent)
        agents_from_repo2 = repo2.get_agents()

        assert len(agents_from_repo2) == 1
        assert agents_from_repo2[0].name == sample_agent.name

    def test_clear_functionality(self, sample_agent):
        """Test that clear() removes all agents."""
        repo = MockAgentRepository()

        # Add agents
        repo.create_agent(sample_agent)
        agent2 = Agent(**sample_agent.model_dump())
        agent2.name = "Second Bot"
        repo.create_agent(agent2)

        assert len(repo.agents) == 2

        # Clear
        repo.clear()

        assert len(repo.agents) == 0
        assert repo.get_agents() == []

    def test_agent_data_integrity(self, sample_agent):
        """Test that agent data is preserved correctly."""
        repo = MockAgentRepository()

        repo.create_agent(sample_agent)
        retrieved = repo.get_agents()[0]

        # Verify all fields are preserved
        assert retrieved.name == sample_agent.name
        assert retrieved.description == sample_agent.description
        assert retrieved.prompt == sample_agent.prompt
        assert retrieved.corpa == sample_agent.corpa
        assert len(retrieved.roles) == len(sample_agent.roles)
        assert retrieved.llm_model == sample_agent.llm_model
        assert retrieved.llm_temperature == sample_agent.llm_temperature
        assert retrieved.llm_max_tokens == sample_agent.llm_max_tokens
        assert retrieved.retrieval_method == sample_agent.retrieval_method
        assert retrieved.status == sample_agent.status

    def test_agent_independence(self, sample_agent):
        """Test that modifying retrieved agent doesn't affect stored data."""
        repo = MockAgentRepository()

        repo.create_agent(sample_agent)
        retrieved = repo.get_agents()[0]

        # Modify retrieved agent
        original_name = retrieved.name
        retrieved.name = "Modified Name"

        # Get agent again
        retrieved_again = repo.get_agents()[0]

        # Original should be unchanged
        assert retrieved_again.name == original_name


# TODO: Add MongoDBAgentRepository tests!


class TestFactoryIntegration:
    """Test that the factory returns the correct repository based on configuration."""

    def test_factory_returns_agent_repository(self):
        """Test that factory returns a valid AgentRepository."""
        repo = get_agent_repository()

        # Should have all required methods
        assert hasattr(repo, "create_agent")
        assert hasattr(repo, "get_agents")
        assert hasattr(repo, "get_agent_by_id")
        assert hasattr(repo, "is_reachable")

        # Should be reachable
        assert repo.is_reachable() is True

    def test_factory_consistency(self):
        """Test that factory returns consistent repository type."""
        repo1 = get_agent_repository()
        repo2 = get_agent_repository()

        assert repo1 is repo2  # Should be singleton
