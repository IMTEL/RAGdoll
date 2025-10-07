"""MongoDB implementation for agent storage."""

from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent
from src.rag_service.repositories.agent.base import AgentRepository


config = Config()


class MongoDBAgentRepository(AgentRepository):
    """MongoDB-backed repository for AI agent configurations.

    Stores and retrieves agent configurations in the 'agents' collection.
    """

    def __init__(self):
        """Initialize MongoDB connection to agents collection."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_AGENT_COLLECTION]

    def create_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration in MongoDB.

        Args:
            agent (Agent): The agent to store

        Returns:
            Agent: The stored agent object (unchanged as Agent doesn't have id)
        """
        agent_dict = agent.model_dump()
        self.collection.insert_one(agent_dict)
        return agent

    def get_agents(self) -> list[Agent]:
        """Retrieve all agent configurations from MongoDB.

        Returns:
            list[Agent]: All stored agents as Agent objects
        """
        agents = list(self.collection.find())
        result = []
        for agent_doc in agents:
            # Remove MongoDB's _id before creating Agent object
            agent_doc.pop("_id", None)
            result.append(Agent(**agent_doc))
        return result

    def get_agent_by_id(self, agent_id: str) -> Agent | None:
        """Retrieve a specific agent by ID.

        Args:
            agent_id (str): The MongoDB ObjectId as a string

        Returns:
            Agent | None: The agent if found, None otherwise
        """
        try:
            agent_doc = self.collection.find_one({"_id": ObjectId(agent_id)})
            if agent_doc:
                agent_doc.pop("_id", None)
                return Agent(**agent_doc)
            return None
        except Exception:
            return None

    def update_agent(self, agent: Agent) -> Agent:
        """Update an existing agent configuration in MongoDB.

        Args:
            agent (Agent): The agent object with updated fields

        Returns:
            Agent: The updated agent object
        """
        if agent.id is None:
            raise ValueError("Agent ID must be set for update.")

        agent_dict = agent.model_dump()
        agent_id = agent_dict.pop("id")

        self.collection.update_one({"_id": ObjectId(agent_id)}, {"$set": agent_dict})

        return agent

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
