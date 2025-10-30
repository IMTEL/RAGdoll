from src.auth.auth_provider.factory import auth_provider_factory
from src.auth.auth_service.auth_service import AuthService
from src.auth.auth_service.base import BaseAuthService
from src.auth.auth_service.open_auth_service import OpenAuthService
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.user.base import UserDao


def auth_service_factory(
    service: str, user_db: UserDao, agent_db: AgentDAO
) -> BaseAuthService:
    match service:
        case "noauth":
            return OpenAuthService(user_db=user_db, agent_db=agent_db)
        case "service":
            return AuthService(
                user_db=user_db, auth_provider_factory=auth_provider_factory
            )
