from bson import ObjectId
from pymongo import MongoClient

from src.config import Config
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


config = Config()


class MongoDBUserDao(UserDao):
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_USER_COLLECTION]

    def set_user(self, user: User) -> User:
        if not user.id:
            # Create new user
            result = self.collection.insert_one(user.model_dump())
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
                user_doc.pop("_id", None)
                return User(**user_doc)
            return None
        except Exception:
            return None

    def get_user_by_provider(
        self, auth_provider: str, provider_user_id: str
    ) -> User | None:
        try:
            user_doc = self.collection.find_one(
                {"auth_provider": auth_provider, "provider_user_id": provider_user_id}
            )
            if user_doc:
                user_doc.pop("_id", None)
                return User(**user_doc)
            return None
        except Exception:
            return None

    def is_reachable(self) -> bool:
        try:
            self.client.admin.command("ping")
            return True
        except Exception as e:
            print(f"Failed to ping MongoDB: {e}")
            return False
