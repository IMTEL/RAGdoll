from src.auth.auth_provider.base import AuthProvider
from src.auth.auth_provider.dev_auth_provider import DevAuthProvider
from src.auth.auth_provider.google_auth_provider import GoogleAuthProvider
from src.config import Config
from src.rag_service.dao.user.base import UserDao


def auth_provider_factory(provider: str, user_db: UserDao) -> AuthProvider:
    config = Config()
    match provider:
        case "google":
            return GoogleAuthProvider(
                web_client_id=config.GOOGLE_CLIENT_ID, domain_name=" ", user_db=user_db
            )
        case "dev":
            if config.ENV != "dev":
                raise ValueError("Cannot login as dev when not in dev environment")
            return DevAuthProvider(user_db=user_db)
        case _:
            print("no provider mached")
            raise ValueError("Unvalid provider : " + provider)
