"""Mock implementation of AgentRepository for testing."""

from src.models.agent import Agent
from src.rag_service.repositories.base import AgentRepository


class MockAgentRepository(AgentRepository):
    """In-memory mock implementation of AgentRepository for testing.

    Stores agent configurations in memory without requiring database connections.
    """

    def __init__(self):
        """Initialize empty agent storage."""
        self.agents = []

    def create_agent(self, agent: Agent) -> dict:
        """Store a new agent configuration in memory.

        Args:
            agent (Agent): The agent to store

        Returns:
            dict: The stored agent with a generated _id
        """
        agent_dict = agent.model_dump()
        agent_dict["_id"] = str(len(self.agents) + 1)
        self.agents.append(agent_dict)
        return agent_dict

    def get_agents(self) -> list[dict]:
        """Retrieve all stored agent configurations.

        Returns:
            list[dict]: All agents stored in memory
        """
        return self.agents

    def is_reachable(self) -> bool:
        """Check if repository is reachable.

        Returns:
            bool: Always True for in-memory storage
        """
        return True

    def clear(self):
        """Clear all stored agents. Useful for test cleanup."""
        self.agents.clear()
