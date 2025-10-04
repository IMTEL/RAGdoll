from abc import ABC, abstractmethod

from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent


config = Config()


class AgentDatabase(ABC):
    """Abstract base class for agent storage backends.

    Subclasses must implement methods for creating and retrieving agents.
    """

    @abstractmethod
    def create_agent(self, agent: Agent) -> dict:
        """Store a new agent in the backend.

        Args:
            agent (Agent): The agent to store.

        Returns:
            dict: The stored agent as a dictionary (may include backend-specific fields).
        """

    @abstractmethod
    def get_agents(self) -> list[dict]:
        """Retrieve all agents from the backend.

        Returns:
            List[dict]: A list of agent dictionaries.
        """


class MongoAgentDatabase(AgentDatabase):
    """MongoDB implementation of AgentDatabase.

    Stores agents in the 'agents' collection.
    """

    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db["agents"]

    def create_agent(self, agent: Agent) -> dict:
        agent_dict = agent.model_dump()
        result = self.collection.insert_one(agent_dict)
        agent_dict["_id"] = str(result.inserted_id)
        return agent_dict

    def get_agents(self) -> list[dict]:
        agents = list(self.collection.find())
        for agent in agents:
            agent["_id"] = str(agent["_id"])
        return agents


class MockAgentDatabase(AgentDatabase):
    """In-memory mock implementation of AgentDatabase for testing purposes."""

    def __init__(self):
        self.agents = []

    def create_agent(self, agent: Agent) -> dict:
        agent_dict = agent.model_dump()
        agent_dict["_id"] = str(len(self.agents) + 1)
        self.agents.append(agent_dict)
        return agent_dict

    def get_agents(self) -> list[dict]:
        return self.agents


def get_agent_database() -> AgentDatabase:
    """Get the database to use.

    Returns:
        AgentDatabase: The configured agent storage backend.
    """
    match config.RAG_DATABASE_SYSTEM.lower():
        case "mock":
            return MockAgentDatabase()
        case "mongodb":
            return MongoAgentDatabase()
        case _:
            raise ValueError("Invalid database type")
