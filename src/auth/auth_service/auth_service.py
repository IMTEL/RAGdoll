import logging
from collections.abc import Callable

from fastapi import HTTPException
from fastapi_jwt_auth import AuthJWT

from src.auth.auth_provider.base import AuthProvider
from src.auth.auth_service.base import BaseAuthService
from src.config import Config
from src.models.users.user import User
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
        self._default_user = None
        
        # Log authentication status on startup
        if self.config.DISABLE_AUTH:
            logger.warning("🔓 AUTHENTICATION DISABLED - Using default user for all requests")
        else:
            logger.info("🔒 Authentication enabled")

    def _get_or_create_default_user(self) -> User:
        """Get or create the default user when authentication is disabled."""
        if self._default_user is not None:
            return self._default_user
            
        # Try to find existing default user
        try:
            user = self.user_db.get_user_by_provider_user_id("system", "default_user")
            if user:
                logger.info(f"Found existing default user: {user.id}")
                self._default_user = user
                return user
        except Exception:
            pass
        
        # Create new default user
        user = User(
            id=None,
            auth_provider="system",
            provider_user_id="default_user",
            name="Default User",
            email="default@local.dev",
            picture=None,
            owned_agents=[],
            api_keys=[]
        )
        
        saved_user = self.user_db.set_user(user)
        logger.info(f"Created default user: {saved_user.id}")
        self._default_user = saved_user
        return saved_user
    
    def login_user(self, token: str, provider: str) -> str:
        # If authentication is disabled, return default user
        if self.config.DISABLE_AUTH:
            logger.info("Login bypassed - using default user")
            user = self._get_or_create_default_user()
            return user.id
            
        logger.info("Logging in user")
        auth_provider: AuthProvider = self.auth_provider_factory(provider, self.user_db)
        user = auth_provider.get_authenticated_user(token)
        if user is None:
            logger.error("Did not manage to find or create user")
            raise ValueError("Login failed")
        return user.id

    def auth(self, authorize: AuthJWT | None, agent_id: str):
        # If authentication is disabled, allow all access
        if self.config.DISABLE_AUTH:
            logger.debug(f"Auth check bypassed for agent {agent_id}")
            return
            
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
        # If authentication is disabled, return default user
        if self.config.DISABLE_AUTH:
            logger.debug("Returning default user (auth disabled)")
            return self._get_or_create_default_user()
            
        if authorize is None:
            logger.warning("No authorization provided")
            raise HTTPException(status_code=401, detail="Unauthorized edit of agent")
        authorize.jwt_required()
        user_id = authorize.get_jwt_subject()
        user = self.user_db.get_user_by_id(user_id)
        if user is None:
            logger.warning(f"User with userId: {user_id} does not exist")
            raise HTTPException(
                status_code=404, detail="Invalid user, could not find user"
            )
        return user
