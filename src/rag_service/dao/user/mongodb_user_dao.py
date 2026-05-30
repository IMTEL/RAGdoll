import logging
import re

from bson import ObjectId
from pymongo import ASCENDING, MongoClient

from src.config import Config
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


config = Config()
logger = logging.getLogger(__name__)


class MongoDBUserDao(UserDao):
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_USER_COLLECTION]
        self._create_indexes()

    def _create_indexes(self) -> None:
        try:
            self.collection.create_index([("id", ASCENDING)], unique=True, sparse=True)
            self.collection.create_index([("email", ASCENDING)], sparse=True)
            self.collection.create_index(
                [("auth_provider", ASCENDING), ("provider_user_id", ASCENDING)],
                unique=True,
            )
        except Exception as e:
            logger.warning(f"Could not create user indexes: {e}")

    def _user_from_mongo(self, user_doc: dict) -> User:
        user_doc = dict(user_doc)
        user_doc.pop("_id", None)
        return User(**user_doc)

    def set_user(self, user: User) -> User:
        if not user.id:
            existing = self.get_user_by_provider(
                user.auth_provider, user.provider_user_id
            )
            if existing:
                user.id = existing.id
                user.owned_agents = user.owned_agents or existing.owned_agents
                user.collaborating_agents = (
                    user.collaborating_agents or existing.collaborating_agents
                )
                user.api_keys = user.api_keys or existing.api_keys
                return self.set_user(user)

            # Create new user
            user_doc = user.model_dump()
            user_doc.pop("id", None)
            result = self.collection.insert_one(user_doc)
            user.id = str(result.inserted_id)

            # Update the document with the string ID
            self.collection.update_one(
                {"_id": ObjectId(user.id)},
                {"$set": {"id": user.id}},
            )
            return user

        try:
            object_id = ObjectId(user.id)
        except Exception as e:
            raise ValueError(
                f"User ID '{user.id}' is not a valid MongoDB ObjectId."
            ) from e

        result = self.collection.update_one(
            {"_id": object_id}, {"$set": user.model_dump()}
        )

        if result.matched_count == 0:
            raise ValueError(f"User with ID {user.id} not found")

        return user

    def get_user_by_id(self, user_id: str) -> User | None:
        try:
            user_doc = self.collection.find_one({"_id": ObjectId(user_id)})
            if user_doc:
                return self._user_from_mongo(user_doc)
            return None
        except Exception as e:
            logger.warning(f"An exception occured when trying to fetch user : {e}")
            return None

    def get_user_by_provider(
        self, auth_provider: str, provider_user_id: str
    ) -> User | None:
        try:
            user_doc = self.collection.find_one(
                {"auth_provider": auth_provider, "provider_user_id": provider_user_id}
            )
            if user_doc:
                return self._user_from_mongo(user_doc)
            return None
        except Exception as e:
            logger.warning(
                f"An exception occured when trying to fetch user by provider: {e}"
            )
            return None

    def get_user_by_email(self, email: str) -> User | None:
        try:
            user_doc = self.collection.find_one({"email": email})
            if user_doc:
                return self._user_from_mongo(user_doc)
            return None
        except Exception as e:
            logger.warning(
                f"An exception occured when trying to fetch user by email: {e}"
            )
            return None

    def search_users(self, query: str, limit: int = 10) -> list[User]:
        query = query.strip()
        if not query:
            return []

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        user_docs = self.collection.find(
            {
                "$or": [
                    {"name": pattern},
                    {"email": pattern},
                    {"provider_user_id": pattern},
                ]
            }
        ).limit(limit)
        return [self._user_from_mongo(user_doc) for user_doc in user_docs]

    def get_users_with_agent(self, agent_id: str) -> list[User]:
        user_docs = self.collection.find(
            {
                "$or": [
                    {"owned_agents": agent_id},
                    {"collaborating_agents": agent_id},
                ]
            }
        )
        return [self._user_from_mongo(user_doc) for user_doc in user_docs]

    def is_reachable(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            logger.error(f"Failed to ping MongoDB: {e}")
            return False
