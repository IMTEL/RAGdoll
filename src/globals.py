# These are instantiated once an then used in the rest of the application
from src.access_service.base import AbstractAccessService
from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.auth.auth_service.base import BaseAuthService
from src.auth.auth_service.factory import auth_service_factory
from src.config import Config
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.factory import get_agent_dao, get_user_dao
from src.rag_service.dao.user.base import UserDao
import os


user_dao: UserDao = get_user_dao()
agent_dao: AgentDAO = get_agent_dao()
# Use OpenAuthService when DISABLE_AUTH is true
if os.getenv("DISABLE_AUTH", "").lower() == "true":
    from src.auth.auth_service.open_auth_service import OpenAuthService

    auth_service = OpenAuthService(user_dao, agent_dao)
else:
    from src.auth.auth_service.auth_service import AuthService

    auth_service = AuthService(user_dao, agent_dao)
access_service: AbstractAccessService = access_service_factory(
    AccessServiceConfig(Config().ACCESS_SERVICE, agent_dao)
)
