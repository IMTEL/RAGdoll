"""Mock implementation of AgentDAO for testing."""

from copy import deepcopy

from src.models.agent import Agent
from src.rag_service.dao import AgentDAO
from src.utils import singleton


@singleton
class MockAgentDAO(AgentDAO):
    """In-memory mock implementation of AgentDAO for testing.

    Stores agent configurations in memory without requiring database connections.
    Uses singleton pattern to ensure tests can clear shared state.
    """

    def __init__(self):
        """Initialize empty agent storage."""
        self.agents: list[Agent] = []

    def add_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration in memory or updates if the agentID already exists.

        Args:
            agent (Agent): The agent to store

        Returns:
            Agent: A deep copy of the stored agent object
        """
        if not agent.id:
            # Set the agent's id to its index in the list
            agent.id = str(len(self.agents))

            # Store the agent
            self.agents.append(agent)
            return deepcopy(agent)
        # ELSE UPDATE EXISTING AGENT
        try:
            index = int(agent.id)
            if 0 <= index < len(self.agents):
                # UPDATE EXISTING AGENT
                self.agents[index] = agent
            else:
                raise ValueError(f"Agent ID '{agent.id}' does not exist for update.")
        except ValueError as e:
            raise ValueError(
                f"Agent ID '{agent.id}' is not a valid index for update."
            ) from e

        return deepcopy(agent)

    def get_agents(self) -> list[Agent]:
        """Retrieve all stored agent configurations.

        Returns:
            list[Agent]: Deep copies of all agents stored in memory
        """
        return [deepcopy(agent) for agent in self.agents]

    # TODO : Update emulator and test to not be dependant on a id, based on index
    # Breaks tests if used in tests
    def delete_agent_by_id(self, agent_id: str) -> bool:
        try:
            index = int(agent_id)
            if 0 <= index < len(self.agents):
                self.agents.remove(self.agents[index])
            return None
        except (ValueError, IndexError):
            return None

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
        """Check if DAO is reachable.

        Returns:
            bool: Always True for in-memory storage
        """
        return True

    def clear(self):
        """Clear all stored agents. Useful for test cleanup."""
        self.agents.clear()
