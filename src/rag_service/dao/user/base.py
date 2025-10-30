from abc import ABC, abstractmethod

from src.models.users.user import User


class UserDao(ABC):
    @abstractmethod
    def set_user(self, agent: User) -> User:
        """Store a new agent configuration and updates if the agent already exists."""

    @abstractmethod
    def get_user_by_id(self, agent_id: str) -> User | None:
        """Retrieve a specific agent by ID."""

    @abstractmethod
    def get_user_by_provider(
        self, auth_provider: str, provider_user_id: str
    ) -> User | None:
        """Retrieve a specific agent by provider and provider given ID."""

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible."""
