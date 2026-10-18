"""MongoDB implementation for agent storage."""

import logging
from datetime import datetime

from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent
from src.rag_service.dao.agent.base import AgentDAO
from src.utils.crypto_utils import decrypt_value, encrypt_str


config = Config()
logger = logging.getLogger(__name__)

_PROVIDER_KEY_FIELDS = ("llm_api_key", "embedding_api_key")
# Set on load to flag undecryptable provider keys; never stored.
_LOAD_ONLY_FIELDS = {f"{field}_invalid" for field in _PROVIDER_KEY_FIELDS}


def _agent_from_doc(agent_doc: dict) -> Agent:
    """Builds an Agent from a stored document, decrypting its provider keys.

    A missing key loads as empty. A key that cannot be decrypted (FERNET_KEY
    changed, or its key was rotated out) also loads as empty, with
    ``<field>_invalid`` set so the user can be asked to enter it again; the
    agent itself still loads.
    """
    for field in _PROVIDER_KEY_FIELDS:
        token = agent_doc.get(field) or ""
        agent_doc[field] = ""
        if not token:
            continue
        try:
            agent_doc[field] = decrypt_value(token)
        except ValueError:
            logger.warning(
                "Agent %s: stored %s cannot be decrypted with the configured "
                "FERNET_KEY and must be entered again",
                agent_doc.get("id"),
                field,
            )
            agent_doc[f"{field}_invalid"] = True
    return Agent(**agent_doc)


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
        agent_dict = agent.model_dump(exclude=_LOAD_ONLY_FIELDS)
        for field in _PROVIDER_KEY_FIELDS:
            if agent_dict[field]:
                agent_dict[field] = encrypt_str(agent_dict[field])
            elif agent.id and not agent_dict.get(f"{field}_id"):
                # An empty key on update keeps the stored one, including one
                # that the current FERNET_KEY cannot decrypt. With a stored-key
                # reference set, the literal key is cleared instead.
                agent_dict.pop(field)

        if not agent.id:
            # Create new agent
            result = self.collection.insert_one(agent_dict)

            # Get the inserted document's ID and set it to the agent object
            agent.id = str(result.inserted_id)

            # Update the document with the string ID
            self.collection.update_one(
                {"_id": ObjectId(agent.id)},
                {"$set": {"id": agent.id}},
            )
            return agent
        # Else update existing agent
        else:
            agent_id = agent_dict.pop("id")

            try:
                object_id = ObjectId(agent_id)
            except Exception as e:
                raise ValueError(
                    f"Agent ID '{agent_id}' is not a valid MongoDB ObjectId."
                ) from e

            result = self.collection.update_one(
                {"_id": object_id}, {"$set": agent_dict}
            )
            if result.matched_count == 0:
                # If agentid is not found:
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
            agent_doc["id"] = str(agent_doc.pop("_id"))
            result.append(_agent_from_doc(agent_doc))
        return result

    def delete_agent_by_id(self, agent_id: str) -> bool:
        """Delete a specific agent by ID.

        Args:
            agent_id (str): The MongoDB ObjectId as a string

        Returns:
            bool: True if the agent was deleted successfully, False otherwise
        """
        try:
            result = self.collection.delete_one({"_id": ObjectId(agent_id)})
            return result.deleted_count > 0
        except Exception:
            return False

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
                agent_doc["id"] = str(agent_doc.pop("_id"))
                return _agent_from_doc(agent_doc)
            return None
        except Exception:
            return None

    def touch_access_key(self, agent_id: str, key_id: str, when: datetime) -> None:
        """Set ``last_use`` of one access key without rewriting the agent."""
        self.collection.update_one(
            {"_id": ObjectId(agent_id), "access_key.id": key_id},
            {"$set": {"access_key.$.last_use": when}},
        )

    def is_reachable(self) -> bool:
        """Verify MongoDB connection health.

        Returns:
            bool: True if connection is active
        """
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            print(f"Failed to ping MongoDB: {e}")  # TODO: use logging
            return False
