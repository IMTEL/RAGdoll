# These are instantiated once an then used in the rest of the application
import os

from src.access_service.base import AbstractAccessService
from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.auth.auth_service.base import BaseAuthService
from src.auth.auth_service.factory import auth_service_factory
from src.config import Config
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.factory import get_agent_dao, get_user_dao
from src.rag_service.dao.user.base import UserDao


user_dao: UserDao = get_user_dao()
agent_dao: AgentDAO = get_agent_dao()
config = Config()
auth_service_name = (
    "noauth" if os.getenv("DISABLE_AUTH", "").lower() == "true" else config.AUTH_SERVICE
)
auth_service: BaseAuthService = auth_service_factory(
    auth_service_name, user_dao, agent_dao
)
access_service: AbstractAccessService = access_service_factory(
    AccessServiceConfig(config.ACCESS_SERVICE, agent_dao)
)
