import random

from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.models.agent import Agent
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


config = Config()


class MockUserDao(UserDao):

    def __init__(self):
        self.users : list[User] = []

    def get_user(self, user_id: str) -> User | None:
        for user in self.users:
            if (user.id == user_id):
                return user
        return None


    def set_user(self, user: User) -> User:
        existing_user = self.get_user(user.id)
        user.id = str(random.randrange(0,1000000))
        if (existing_user is not None):
            user.id = existing_user.id
            self.users.remove(existing_user)
        self.users.append(user)
        return user
        

    def get_user_by_id(self, user_id: str) -> Agent | None:
        return self.get_user(user_id)

        
    def get_user_by_provider(self,auth_provider : str ,provider_user_id: str) -> Agent | None:
        for user in self.users:
            if (user.auth_provider == auth_provider and user.provider_user_id == provider_user_id):
                return user
        return None

    def is_reachable(self) -> bool:
        return True