"""Tests for the template endpoints.

The DAO and the current user are replaced through FastAPI dependency
overrides, so the tests need neither MongoDB nor a login.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.users.user import User
from src.routes.templates import get_current_user, get_dao
from tests.mocks.mock_template_dao import MockTemplateDAO


client = TestClient(app)

ALICE = User(id="alice", auth_provider="test", provider_user_id="alice")
BOB = User(id="bob", auth_provider="test", provider_user_id="bob")

INTERVIEWER = {
    "type": "interviewer",
    "name": "Butikksjef",
    "description": "Vennlig og direkte.",
    "content": {"persona": "Intervjueren er butikksjef.", "focus": ["Service"]},
}


@pytest.fixture(autouse=True)
def overrides():
    """Use the mock DAO and log in as Alice by default."""
    repo = MockTemplateDAO()
    repo.clear()
    state = {"user": ALICE}
    app.dependency_overrides[get_dao] = lambda: repo
    app.dependency_overrides[get_current_user] = lambda: state["user"]

    yield state

    app.dependency_overrides.pop(get_dao, None)
    app.dependency_overrides.pop(get_current_user, None)
    repo.clear()


def _create(**fields) -> dict:
    response = client.post("/templates", json={**INTERVIEWER, **fields})
    assert response.status_code == 201, response.text
    return response.json()


class TestTemplateEndpoints:
    """Tests for /templates endpoints."""

    def test_list_empty(self):
        response = client.get("/templates")

        assert response.status_code == 200
        assert response.json() == []

    def test_create_sets_id_and_owner(self):
        created = _create(name="  Butikksjef  ")

        assert created["id"]
        assert created["owner_id"] == "alice"
        assert created["name"] == "Butikksjef"
        assert created["content"] == INTERVIEWER["content"]

    def test_list_filters_by_type(self):
        _create()
        _create(type="other", name="Annen")

        response = client.get("/templates", params={"type": "interviewer"})

        assert [t["name"] for t in response.json()] == ["Butikksjef"]

    def test_get_update_and_delete(self):
        created = _create()
        url = f"/templates/{created['id']}"

        assert client.get(url).json()["name"] == "Butikksjef"

        updated = client.put(url, json={**INTERVIEWER, "name": "Kjøpmann"})
        assert updated.status_code == 200
        assert updated.json()["name"] == "Kjøpmann"
        assert updated.json()["created_at"] == created["created_at"]

        assert client.delete(url).status_code == 204
        assert client.get(url).status_code == 404

    def test_other_users_template_is_not_found(self, overrides):
        created = _create()
        url = f"/templates/{created['id']}"
        overrides["user"] = BOB

        assert client.get(url).status_code == 404
        assert client.put(url, json=INTERVIEWER).status_code == 404
        assert client.delete(url).status_code == 404
        assert client.get("/templates").json() == []

        overrides["user"] = ALICE
        assert client.get(url).status_code == 200

    def test_rejects_empty_name(self):
        assert (
            client.post("/templates", json={**INTERVIEWER, "name": " "}).status_code
            == 422
        )

    def test_rejects_unknown_fields(self):
        body = {**INTERVIEWER, "owner_id": "bob"}

        assert client.post("/templates", json=body).status_code == 422

    def test_rejects_too_large_content(self):
        body = {**INTERVIEWER, "content": {"persona": "x" * (65 * 1024)}}

        assert client.post("/templates", json=body).status_code == 422
