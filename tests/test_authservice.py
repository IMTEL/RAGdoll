from fastapi import HTTPException
import pytest
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from src.auth.auth_service.auth_service import AuthService
from tests.mocks.mock_auth_provider_factory import mock_auth_provider_factory
from tests.mocks.mock_user_dao import MockUserDao


def get_services():
    user_dao = MockUserDao()
    auth_service = AuthService(user_dao, mock_auth_provider_factory)
    return auth_service, user_dao


class Settings(BaseModel):
    authjwt_secret_key: str = "supersecret-test-key-no-one-knows"


@AuthJWT.load_config
def get_config():
    return Settings()


def create_token(user_id: str):
    """Create a JWT with a specific subject."""
    authorize = AuthJWT()
    authorize._token = authorize.create_access_token(subject=user_id)
    authorize._token_location = "headers"
    return authorize


@pytest.mark.unit
def test_creation_of_services():
    get_services()


@pytest.mark.unit
def test_login_creating_user():
    auth_service, user_dao = get_services()
    user_id = auth_service.login_user("yehaw", "mock")
    user = user_dao.get_user_by_id(user_id)
    assert user.provider_user_id == "yehaw"
    assert user.auth_provider == "mock"


@pytest.mark.unit
def test_get_authenticated_user():
    auth_service, user_dao = get_services()
    user_id = auth_service.login_user("yehaw", "mock")

    user = user_dao.get_user_by_id(user_id)
    assert user.provider_user_id == "yehaw"
    assert user.auth_provider == "mock"

    authorize = create_token(user_id)
    assert auth_service.get_authenticated_user(authorize) == user


@pytest.mark.unit
def test_auth_no_user():
    auth_service, user_dao = get_services()
    authorize = create_token("no user")
    with pytest.raises(HTTPException):
        auth_service.get_authenticated_user(authorize)


@pytest.mark.unit
def test_auth():
    auth_service, user_dao = get_services()
    user_id = auth_service.login_user("yehaw", "mock")

    user = user_dao.get_user_by_id(user_id)
    assert user.provider_user_id == "yehaw"
    assert user.auth_provider == "mock"
    user.owned_agents = ["test_agent_id"]
    user_dao.set_user(user)

    authorize = create_token(user_id)
    auth_service.auth(authorize,"test_agent_id")
