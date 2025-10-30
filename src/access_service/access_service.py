import base64
import logging
import secrets
import uuid
from datetime import datetime

from src.access_service.base import AbstractAccessService
from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.rag_service.dao.agent.base import AgentDAO


logger = logging.getLogger(__name__)


class AccessService(AbstractAccessService):
    def __init__(self, agent_database: AgentDAO):
        self.agent_database = agent_database

    def try_get_agent(self, agent_id: str):
        agent = self.agent_database.get_agent_by_id(agent_id)
        if agent is None:
            logger.warning(f"Agent no found : {agent_id}")
            raise ValueError(f"No agent found by id : {agent_id}")
        return agent

    def get_unique_access_key_id(self, agent: Agent):
        tries = 0
        existing_keys = {ak.id for ak in agent.access_key}
        while tries < 50:
            new_id = uuid.uuid4().hex[:8]  # 8 Byte should be enough
            if new_id not in existing_keys:
                return new_id
            tries += 1
        raise ValueError("Could not generate a new key!")

    def generate_accesskey(
        self, name: str, expiry_date: datetime | None, agent_id: str
    ) -> AccessKey:
        if (expiry_date is not None) and expiry_date < datetime.now():
            raise ValueError("Expiry date cannot be in the past")

        agent = self.try_get_agent(agent_id)
        key = secrets.token_bytes(32)  # AES-256 32byte secret
        key_str = base64.urlsafe_b64encode(key).decode("ascii")
        access_key = AccessKey(
            id=self.get_unique_access_key_id(agent),
            key=key_str,
            name=name,
            expiry_date=expiry_date,
            created=datetime.now(),
            last_use=None,
        )
        agent.access_key.append(access_key)
        self.agent_database.add_agent(agent)
        return access_key

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        try:
            agent = self.try_get_agent(agent_id)
        except Exception as e:
            logger.warning(f"An exception occured while trying to fetch agent {e}")
            return False

        access_keys: list[AccessKey] = agent.access_key

        for ak in access_keys:
            if ak.key == access_key:
                if ak.expiry_date is None:
                    return True
                if datetime.now() < ak.expiry_date:
                    return True
                logger.warning(f"Access key with name : {ak.name} has expired")
                return False
        logger.warning("No matching access key found in agent")
        return False

    def get_access_key_by_id(
        self, agent: Agent, access_key_id: str
    ) -> AccessKey | None:
        for ak in agent.access_key:
            if ak.id == access_key_id:
                return ak
        return None

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        agent = self.try_get_agent(agent_id)
        key = self.get_access_key_by_id(agent, access_key_id)
        if key is None:
            return False
        agent.access_key.remove(key)
        self.agent_database.add_agent(agent)
        return True
