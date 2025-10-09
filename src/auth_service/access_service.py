import secrets
import uuid
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime

from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.rag_service.repositories.agent.base import AgentRepository


class AbstractAccessService:
    @abstractmethod
    def generate_accesskey(
        self, name: str, expiery_date: datetime, agent_id: str
    ) -> AccessKey:
        """Generates an accesskey and adds it into the database, returns the generated key."""

    @abstractmethod
    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        """Revokes the spesified token."""

    @abstractmethod
    def authenticate(self, agent_id: str, access_key: str) -> bool:
        """Authenticates the use of an accesskey."""


class DummyAccessService(AbstractAccessService):
    def generate_accesskey(
        self, name: str, expiery_date: datetime, agent_id: str
    ) -> AccessKey:
        return AccessKey(
            key="", name=name, expiery_date=expiery_date, created=datetime.now()
        )

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        return True

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        return True


class AccessService(AbstractAccessService):
    def __init__(self, agent_database):
        self.agent_database: AgentRepository = agent_database

    def try_get_agent(self, agent_id: str):
        agent = self.agent_database.get_agent_by_id(agent_id)
        if agent == None:
            raise ValueError(f"No agent found by id : {agent_id}")
        return agent

    def get_uniqe_access_key_id(self, agent: Agent):
        tries = 0
        existing_keys = {ak.id for ak in agent.access_key}
        while tries < 50:
            new_id = str(uuid.uuid4().hex[:8])  # 8 Byte should be enough
            if new_id not in existing_keys:
                return new_id
            tries += 1
        raise ValueError("Could not generate a new key!")

    def generate_accesskey(
        self, name: str, expiery_date: datetime, agent_id: str
    ) -> AccessKey:
        if expiery_date < datetime.now():
            raise ValueError("Expiery date cannot be in the past")

        agent = self.try_get_agent(agent_id)

        key = secrets.token_bytes(32)  # AES-256 32byte secret
        access_key = AccessKey(
            id=self.get_uniqe_access_key_id(agent),
            key=key,
            name=name,
            expiery_date=expiery_date,
            created=datetime.now(),
        )
        agent.access_key.append(access_key)
        self.agent_database.update_agent(agent)
        return AccessKey

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        agent = self.try_get_agent(agent_id)

        access_keys: list[AccessKey] = agent.access_key

        for ak in access_keys:
            if ak.key == access_key:
                if ak.expiery_date == None:
                    return True
                return datetime.now() > ak.expiery_date

        return False

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        agent = self.try_get_agent(agent_id)

        for ak in agent.access_key:
            if ak.id == access_key_id:
                agent.access_key.remove(ak)
                self.agent_database.update_agent(agent)
                return True
        return False


@dataclass
class AccessServiceConfig:
    type: str
    database: AgentRepository | None


def access_service_factory(config: AccessServiceConfig):
    match config.type.lower():
        case "dummy":
            return DummyAccessService()
        case "service":
            return AccessService(config.database)
        case _:
            raise ValueError(f"AccessService {config.type} does not exist")
