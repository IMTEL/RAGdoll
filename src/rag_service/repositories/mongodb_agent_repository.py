"""MongoDB implementation for agent storage."""

from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent
from src.rag_service.repositories.base import AgentRepository


config = Config()


class MongoDBAgentRepository(AgentRepository):
    """MongoDB-backed repository for AI agent configurations.

    Stores and retrieves agent configurations in the 'agents' collection.
    """

    def __init__(self):
        """Initialize MongoDB connection to agents collection."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db["agents"]

    def create_agent(self, agent: Agent) -> dict:
        """Store a new agent configuration in MongoDB.

        Args:
            agent (Agent): The agent to store

        Returns:
            dict: The stored agent including MongoDB's _id
        """
        agent_dict = agent.model_dump()
        result = self.collection.insert_one(agent_dict)
        agent_dict["_id"] = str(result.inserted_id)
        return agent_dict

    def get_agents(self) -> list[dict]:
        """Retrieve all agent configurations from MongoDB.

        Returns:
            list[dict]: All stored agents with string-converted _id fields
        """
        agents = list(self.collection.find())
        for agent in agents:
            agent["_id"] = str(agent["_id"])
        return agents

    def is_reachable(self) -> bool:
        """Verify MongoDB connection health.

        Returns:
            bool: True if connection is active
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            print(f"Failed to ping MongoDB: {e}")
            return False
