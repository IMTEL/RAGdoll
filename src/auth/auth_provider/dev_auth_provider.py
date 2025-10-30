from src.auth.auth_provider.base import AuthProvider
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao


class DevAuthProvider(AuthProvider):
    """Authenticates the test dev user when in development Environment."""

    def __init__(self, user_db: UserDao):
        self.user_db = user_db

    def get_authenticated_user(self, token: str) -> User | None:
        user = self.user_db.get_user_by_provider(DevAuthProvider.get_provider(), token)
        if user is None:
            return self.user_db.set_user(
                User(
                    auth_provider=DevAuthProvider.get_provider(),
                    provider_user_id="dev",
                    name="dev",
                )
            )
        return user

    @staticmethod
    def get_provider() -> str:
        return "dev"
