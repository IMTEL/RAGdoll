import random

from src.config import Config
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


config = Config()


class MockUserDao(UserDao):
    def __init__(self):
        self.users: list[User] = []

    def get_user(self, user_id: str) -> User | None:
        for user in self.users:
            if user.id == user_id:
                return user.model_copy()
        return None

    def set_user(self, user: User) -> User:
        existing_user = self.get_user(user.id)
        user.id = str(random.randrange(0, 1000000))
        if existing_user is not None:
            user.id = existing_user.id
            self.users.remove(existing_user)
        self.users.append(user)
        return user

    def get_user_by_id(self, user_id: str) -> User | None:
        return self.get_user(user_id)

    def get_user_by_provider(
        self, auth_provider: str, provider_user_id: str
    ) -> User | None:
        for user in self.users:
            if (
                user.auth_provider == auth_provider
                and user.provider_user_id == provider_user_id
            ):
                return user
        return None

    def get_user_by_email(self, email: str) -> User | None:
        for user in self.users:
            if user.email == email:
                return user
        return None

    def search_users(self, query: str, limit: int = 10) -> list[User]:
        query = query.lower().strip()
        if not query:
            return []

        matches = []
        for user in self.users:
            fields = [user.name or "", user.email or "", user.provider_user_id]
            if any(query in field.lower() for field in fields):
                matches.append(user)
            if len(matches) >= limit:
                break
        return matches

    def get_users_with_agent(self, agent_id: str) -> list[User]:
        return [
            user
            for user in self.users
            if agent_id in user.owned_agents or agent_id in user.collaborating_agents
        ]

    def is_reachable(self) -> bool:
        return True
