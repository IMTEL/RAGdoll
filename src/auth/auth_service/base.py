from abc import abstractmethod

from fastapi_jwt_auth import AuthJWT

from src.models.users.user import User


class BaseAuthService:
    @abstractmethod
    def login_user(self, token: str, provider: str) -> str:
        """Returns the user."""

    @abstractmethod
    def auth(self, authorize: AuthJWT, agent_id: str):
        """Helper function, auhtorizes a user to edit an agent.

        raises HTTP exception if user is not authorized to edit agent
        """

    @abstractmethod
    def get_authenticated_user(self, authorize: AuthJWT) -> User:
        """Checks for a jwt token, authenticates it and returns a user.

        raises an HTTP exception when there is no jwt-token or no user

        """
