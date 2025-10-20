import logging

from google.auth.transport import requests
from google.oauth2 import id_token

from src.auth.auth_provider.base import AuthProvider
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


logger = logging.getLogger(__name__)


class GoogleAuthProvider(AuthProvider):
    def __init__(self, web_client_id: str, domain_name: str, user_db: UserDao):
        self.web_client_id = web_client_id
        self.domain_name = domain_name
        self.user_db = user_db

    def authenticate_user(self, token: str) -> str | None:
        """Returns a valid google user-id, if the token is valid."""
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), self.web_client_id
        )
        # TODO : Bad authentication, check for domain, add user and email fix later

        return idinfo["sub"]

    def get_authenticated_user(self, token: str) -> User | None:
        google_id = self.authenticate_user(token)
        if google_id is None:
            return None
        user = self.user_db.get_user_by_provider(
            GoogleAuthProvider.get_provider(), google_id
        )
        if user is not None:
            return user
        return self.user_db.set_user(
            User(
                provider_user_id=google_id,
                name="default",
                auth_provider=GoogleAuthProvider.get_provider(),
                owned_agents=[],
            )
        )

    @staticmethod
    def get_provider() -> str:
        return "google"
