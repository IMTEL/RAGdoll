from fastapi_jwt_auth import AuthJWT

from src.auth.auth_service.base import BaseAuthService
from src.models.users.user import User
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.user.base import UserDao


"""Ignores Authentication, gives auth to every agent.

This service is used when AUTH_SERVICE=noauth is set in the environment.
It returns a default 'user' account that has access to all agents.
"""


class OpenAuthService(BaseAuthService):
    DEFAULT_USER_ID = "default_user"  # Consistent user ID
    
    def __init__(self, user_db: UserDao, agent_db: AgentDAO):
        self.user_db = user_db
        self.agent_db = agent_db
        self._cached_user = None

    def get_default_user(self) -> User:
        """Get or create the default user with access to all agents.
        
        This user is named 'user' and has access to all agents in the system.
        The user is cached to avoid unnecessary database calls.
        """
        # Try to get existing user from database
        user = self.user_db.get_user_by_id(self.DEFAULT_USER_ID)
        
        if user is None:
            # Create default user if it doesn't exist
            agents = self.agent_db.get_agents()
            agent_ids = [agent.id for agent in agents]
            user = User(
                id=self.DEFAULT_USER_ID,
                name="user",  # Simple name: "user"
                auth_provider="noauth",
                provider_user_id=self.DEFAULT_USER_ID,
                owned_agents=agent_ids,
            )
            user = self.user_db.set_user(user)
        else:
            # Update existing user to own all current agents
            agents = self.agent_db.get_agents()
            agent_ids = [agent.id for agent in agents]
            if set(user.owned_agents) != set(agent_ids):
                user.owned_agents = agent_ids
                user = self.user_db.set_user(user)
        
        return user

    def login_user(self, token: str, provider: str) -> str:
        """Return the default user ID for login."""
        return self.get_default_user().id

    def auth(self, authorize: AuthJWT | None, agent_id: str):
        """Always pass - no authentication required when auth is disabled."""
        return

    def get_authenticated_user(self, authorize: AuthJWT | None) -> User:
        """Return the default user - no JWT validation required."""
        return self.get_default_user()
