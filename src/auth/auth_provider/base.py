from abc import abstractmethod

from src.models.users.user import User


class AuthProvider:
    @abstractmethod
    def get_authenticated_user(self, token: str) -> User | None:
        """Authenticates a token and return a user."""

    @staticmethod
    @abstractmethod
    def get_provider(self, name) -> str:
        """Returns the name of a spesific provider."""
