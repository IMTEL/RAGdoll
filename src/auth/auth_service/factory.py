from src.auth.auth_provider.factory import auth_provider_factory
from src.auth.auth_service.auth_service import AuthService
from src.auth.auth_service.base import BaseAuthService
from src.auth.auth_service.open_auth_service import OpenAuthService
from src.rag_service.dao.factory import get_agent_dao, get_user_dao


def auth_service_factory(service: str) -> BaseAuthService:
    match service:
        case "noauth":
            return OpenAuthService(user_db=get_user_dao(), agent_db=get_agent_dao())
        case "service":
            return AuthService(
                user_db=get_user_dao(), auth_provider_factory=auth_provider_factory
            )
