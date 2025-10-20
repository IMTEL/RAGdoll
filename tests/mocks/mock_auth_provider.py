from abc import abstractmethod

import random

from src.auth.auth_provider.base import AuthProvider
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


class MockAuthProvider(AuthProvider):
    """Authenticates mock user when in development Environment, provider_user_id will be token."""

    def __init__(self, user_db : UserDao):
        self.user_db = user_db

    def get_authenticated_user(self, token : str) -> User | None:
        user = self.user_db.get_user_by_provider(MockAuthProvider.get_provider(),token)
        if (user is not None):
            return user
        return self.user_db.set_user(User(auth_provider=MockAuthProvider.get_provider(),provider_user_id=token,name = "Bob")) 

    @staticmethod
    def get_provider() -> str:
        return "mock"
