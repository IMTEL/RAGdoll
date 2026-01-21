import logging
import os
from collections.abc import Callable

from fastapi import HTTPException
from fastapi_jwt_auth import AuthJWT

from src.auth.auth_provider.base import AuthProvider
from src.auth.auth_service.base import BaseAuthService
from src.config import Config
from src.models.user import User
from src.rag_service.dao.user.base import UserDao


logger = logging.getLogger(__name__)


class AuthService(BaseAuthService):
    def __init__(
        self,
        user_db: UserDao,
        auth_provider_factory: Callable[[str, UserDao], AuthProvider],
    ):
        self.user_db = user_db
        self.auth_provider_factory = auth_provider_factory
        self.config = Config()

    def login_user(self, token: str, provider: str) -> str:
        logger.info("Logging in user")
        auth_provider: AuthProvider = self.auth_provider_factory(provider, self.user_db)
        user = auth_provider.get_authenticated_user(token)
        if user is None:
            logger.error("Did not manage to find or create user")
            raise ValueError("Login failed")
        return user.id

    def auth(self, authorize: AuthJWT | None, agent_id: str):
        if authorize is None:
            logger.warning("No authorization provided")
            raise HTTPException(status_code=401, detail="Unauthorized edit of agent")
        user = self.get_authenticated_user(authorize)
        if agent_id not in user.owned_agents:
            logger.warning(
                f"User tried to access agent they dont own user: {user.id}, agent : {agent_id}"
            )
            raise HTTPException(status_code=401, detail="Unnauthorized edit of agent")

    def get_authenticated_user(self, authorize: AuthJWT | None) -> User:
        if authorize is None:
            logger.warning("No authorization provided")
            raise HTTPException(status_code=401, detail="Unauthorized edit of agent")
        # Demo mode - bypass authentication
        if os.getenv("DISABLE_AUTH", "").lower() == "true":
            # Return a demo user
            return self._get_or_create_demo_user()
        authorize.jwt_required()
        user_id = authorize.get_jwt_subject()
        user = self.user_db.get_user_by_id(user_id)
        if user is None:
            logger.warning(f"User with userId: {user_id} does not exist")
            raise HTTPException(
                status_code=404, detail="Invalid user, could not find user"
            )
        return user

    def _get_or_create_demo_user(self) -> User:
        from src.rag_service.dao.user.user_dao import user_dao

        demo_user = user_dao.get_user_by_email("demo@example.com")
        if not demo_user:
            demo_user = User(
                email="demo@example.com",
                name="Demo User",
                api_keys=[],
            )
            user_dao.set_user(demo_user)
        return demo_user
