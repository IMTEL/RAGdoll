from fastapi_jwt_auth import AuthJWT
import os

from src.auth.auth_service.base import BaseAuthService
from src.models.users.user import User
from src.rag_service.dao.agent.base import AgentDAO
from src.rag_service.dao.user.base import UserDao


"""Ingores Authentication, gives auth to every agent"""


class OpenAuthService(BaseAuthService):
    def __init__(self, user_db: UserDao, agent_db: AgentDAO):
        self.user_db = user_db
        self.agent_db = agent_db

    def get_mock_user(self) -> User:
        agents = self.agent_db.get_agents()
        agent_ids = [agent.id for agent in agents]
        user = User(
            name="mock",
            auth_provider="mock",
            provider_user_id="mock",
            owned_agents=agent_ids,
        )
        user = self.user_db.set_user(user)
        return user

    def login_user(self, token: str, provider: str) -> str:
        return self.get_mock_user().id

    """ Always pass """

    def auth(self, authorize: AuthJWT | None, agent_id: str):
        return

    def get_authenticated_user(self, authorize: AuthJWT | None) -> User:
        return self.get_mock_user()


# Use OpenAuthService when DISABLE_AUTH is true
if os.getenv("DISABLE_AUTH", "").lower() == "true":
    from src.auth.auth_service.open_auth_service import OpenAuthService

    auth_service = OpenAuthService(user_dao, agent_dao)
else:
    # ...existing auth service initialization...
    from src.auth.auth_service.auth_service import AuthService

    auth_service = AuthService(user_dao, agent_dao)
