
from dataclasses import dataclass
from src.models.accesstoken import AccessKey
from datetime import datetime
import secrets

from src.rag_service.agent_dao import AgentDatabase
from src.rag_service.dao import get_database

class AbstractAccessService():
    @abstractmethod
    def generate_accesstoken(self, name: str, expiery_date : datetime, agent_id : str) -> AccessKey:
        """
        Generates an accesskey and adds it into the database, returns the generated key
        """
        pass

    @abstractmethod
    def revoke_key(self, access_key : str) -> bool:
        """
        Revokes the spesified token
        """
        pass

    @abstractmethod
    def authenticate(agent_id: str, access_key : str) -> bool:
        """
        Authenticates the use of an accesskey
        """
        pass

class DummyAccessService(AbstractAccessService):
    def generate_accesstoken(name: str, expiery_date : datetime, agent_id : str) -> AccessKey:
        return AccessKey(key="",name=name,expiery_date=expiery_date)

    def authenticate(agent_id: str, access_key : str) -> bool:
        return True
    
    def revoke_key(self, access_key) -> bool:
        return False
    

class AccessService(AbstractAccessService):
    def __init__(self, agent_database):
        self.agent_database : AgentDatabase = agent_database

    def generate_accesstoken(self, name: str, expiery_date : datetime, agent_id : str) -> AccessKey:
        if (expiery_date < datetime.now()):
            raise ValueError("Expiery date cannot be in the past")
        
        key = secrets.token_bytes(32) # AES-256 32byte secret

        access_key = AccessKey(key,name,expiery_date)
        # TODO : Add to agent

    def authenticate(agent_id: str, access_key : str) -> bool:
        # TODO : Check if its in agent

        return True
    
    def revoke_key(self, access_key) -> bool:
        return super().revoke_key(access_key)


@dataclass
class AccessServiceConfig:
    type: str
    database: AgentDatabase | None


def access_service_factory(config : AccessServiceConfig):
    match (config.type.lower()):
        case "dummy":
            return DummyAccessService()
        case "service":
            return AccessService(config.database)
        case _:            
            raise ValueError(f"AccessService {config.type} does not exist")

    
