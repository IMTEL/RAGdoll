"""Tests for the user settings endpoints.

The DAO and the current user are replaced through FastAPI dependency
overrides, so the tests need neither MongoDB nor a login.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.users.user import User
from src.routes.user_settings import get_current_user, get_dao
from tests.mocks.mock_user_dao import MockUserDao


client = TestClient(app)


@pytest.fixture(autouse=True)
def overrides():
    """Use a fresh mock DAO and a stored user for each test."""
    repo = MockUserDao()
    user = repo.set_user(User(auth_provider="test", provider_user_id="alice"))
    app.dependency_overrides[get_dao] = lambda: repo
    app.dependency_overrides[get_current_user] = lambda: repo.get_user_by_id(user.id)

    yield repo

    app.dependency_overrides.pop(get_dao, None)
    app.dependency_overrides.pop(get_current_user, None)


class TestUserSettingsEndpoints:
    """Tests for /user-settings endpoints."""

    def test_empty_before_any_are_saved(self):
        response = client.get("/user-settings/test-app")

        assert response.status_code == 200
        assert response.json() == {}

    def test_saved_settings_are_returned(self):
        settings = {"language": "nb", "showHints": False}

        assert client.put("/user-settings/test-app", json=settings).status_code == 200
        assert client.get("/user-settings/test-app").json() == settings

    def test_namespaces_do_not_overwrite_each_other(self):
        client.put("/user-settings/app-one", json={"theme": "dark"})
        client.put("/user-settings/app-two", json={"theme": "light"})

        assert client.get("/user-settings/app-one").json() == {"theme": "dark"}

    def test_rejects_invalid_namespace(self):
        assert client.get("/user-settings/Not_Valid").status_code == 422

    def test_rejects_settings_that_are_not_an_object(self):
        assert client.put("/user-settings/test-app", json=["nb"]).status_code == 422

    def test_rejects_too_large_settings(self):
        response = client.put("/user-settings/test-app", json={"x": "y" * 17 * 1024})

        assert response.status_code == 413
