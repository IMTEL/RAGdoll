"""Mock implementation of AgentRepository for testing."""

from copy import deepcopy

from src.domain.agents import Agent
from src.rag_service.repositories import AgentRepository


class MockAgentRepository(AgentRepository):
    """In-memory mock implementation of AgentRepository for testing.

    Stores agent configurations in memory without requiring database connections.
    Uses singleton pattern to ensure tests can clear shared state.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize empty agent storage (only once for singleton)."""
        if not MockAgentRepository._initialized:
            self.agents: list[Agent] = []
            MockAgentRepository._initialized = True

    def create_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration in memory.

        Args:
            agent (Agent): The agent to store

        Returns:
            Agent: A deep copy of the stored agent object
        """
        # Store a deep copy to prevent external mutations
        self.agents.append(deepcopy(agent))
        return deepcopy(agent)

    def get_agents(self) -> list[Agent]:
        """Retrieve all stored agent configurations.

        Returns:
            list[Agent]: Deep copies of all agents stored in memory
        """
        return [deepcopy(agent) for agent in self.agents]

    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """Retrieve a specific agent by index (using id as index).

        Args:
            agent_id (str): The agent index as string

        Returns:
            Agent | None: A deep copy of the agent if found, None otherwise
        """
        try:
            index = int(agent_id)
            if 0 <= index < len(self.agents):
                return deepcopy(self.agents[index])
            return None
        except (ValueError, IndexError):
            return None

    def is_reachable(self) -> bool:
        """Check if repository is reachable.

        Returns:
            bool: Always True for in-memory storage
        """
        return True

    def clear(self):
        """Clear all stored agents. Useful for test cleanup."""
        self.agents.clear()
