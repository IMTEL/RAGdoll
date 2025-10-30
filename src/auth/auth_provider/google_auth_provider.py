import logging
from dataclasses import dataclass

from google.auth.transport import requests
from google.oauth2 import id_token

from src.auth.auth_provider.base import AuthProvider
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


logger = logging.getLogger(__name__)


@dataclass
class GoogleUserData:
    id: str
    name: str | None
    email: str | None
    picture: str | None


class GoogleAuthProvider(AuthProvider):
    def __init__(self, web_client_id: str, user_db: UserDao):
        self.web_client_id = web_client_id
        self.user_db = user_db

    def authenticate_user(self, token: str) -> GoogleUserData:
        """Returns a valid google user-id, if the token is valid."""
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), self.web_client_id
        )

        user_id = idinfo.get("sub", None)
        if user_id is None:
            raise ValueError("No userid fetched from google!")

        return GoogleUserData(
            user_id,
            idinfo.get("name", None),
            idinfo.get("email", None),
            idinfo.get("picture", None),
        )

    def get_authenticated_user(self, token: str) -> User | None:
        user_data = self.authenticate_user(token)
        user = self.user_db.get_user_by_provider(
            GoogleAuthProvider.get_provider(), user_data.id
        )
        if user is not None:
            return user
        return self.user_db.set_user(
            User(
                provider_user_id=user_data.id,
                name=user_data.name,
                email=user_data.email,
                picture=user_data.picture,
                auth_provider=GoogleAuthProvider.get_provider(),
                owned_agents=[],
            )
        )

    @staticmethod
    def get_provider() -> str:
        return "google"
