"""MongoDB implementation for agent storage."""

from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent
from src.rag_service.dao.agent.base import AgentDAO


config = Config()


class MongoDBAgentDAO(AgentDAO):
    """MongoDB-backed DAO for AI agent configurations.

    Stores and retrieves agent configurations in the 'agents' collection.
    """

    def __init__(self):
        """Initialize MongoDB connection to agents collection."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_AGENT_COLLECTION]

    def add_agent(self, agent: Agent) -> Agent:
        """Store a new agent configuration in MongoDB or updates an existing one.

        Args:
            agent (Agent): The agent to store

        Returns:
            Agent: The stored agent object (unchanged as Agent doesn't have id)
        """
        agent_dict = agent.model_dump()

        if agent.id == "":
            # CREATE NEW AGENT
            result = self.collection.insert_one(agent_dict)

            # Get the inserted document's ID and set it to the agent object
            agent.id = str(result.inserted_id)

            # Update the document with the string ID
            self.collection.update_one(
                {"_id": ObjectId(agent.id)},
                {"$set": {"id": agent.id}},
            )
            return agent
        # ELSE UPDATE EXISTING AGENT
        agent_id = agent_dict.pop("id")

        result = self.collection.update_one({"_id": ObjectId(agent_id)}, {"$set": agent_dict})
        if result.matched_count == 0:
            # IF AGENTID NOT FOUND: 
            raise ValueError(f"Agent with ID {agent_id} not found")
        return agent

    def get_agents(self) -> list[Agent]:
        """Retrieve all agent configurations from MongoDB.

        Returns:
            list[Agent]: All stored agents as Agent objects
        """
        agents = list(self.collection.find())
        result = []
        for agent_doc in agents:
            agent_doc["id"] = str(agent_doc["_id"])
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