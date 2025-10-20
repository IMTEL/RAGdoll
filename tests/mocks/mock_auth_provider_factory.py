
from src.auth.auth_provider.base import AuthProvider
from src.rag_service.dao.user.base import UserDao
from tests.mocks.mock_auth_provider import MockAuthProvider


def mock_auth_provider_factory(provider : str, user_db : UserDao) -> AuthProvider:
    if (provider != "mock"):
        raise ValueError("Provide has to be mock while using mock_auth_provider_factory")
    
    return MockAuthProvider(user_db)