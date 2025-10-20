from src.auth.auth_provider.factory import auth_provider_factory
from src.auth.auth_service.auth_service import AuthService
from src.auth.auth_service.base import BaseAuthService
from src.rag_service.dao.factory import get_user_dao


def auth_service_factory(service: str) -> BaseAuthService:
    match service:
        case "mock":
            # TODO : Add a mock and a no-auth auth-service
            return None
        case "service":
            return AuthService(
                user_db=get_user_dao(), auth_provider_factory=auth_provider_factory
            )
