"""Tests for the job seeker profile endpoints.

The DAO and the current user are replaced through FastAPI dependency
overrides, so the tests need neither MongoDB nor a login.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.models.users.user import User
from src.routes.profiles import get_current_user, get_dao
from tests.mocks.mock_profile_dao import MockProfileDAO


client = TestClient(app)

ALICE = User(id="alice", auth_provider="test", provider_user_id="alice")
BOB = User(id="bob", auth_provider="test", provider_user_id="bob")


@pytest.fixture(autouse=True)
def overrides():
    """Use the mock DAO and log in as Alice by default."""
    repo = MockProfileDAO()
    repo.clear()
    state = {"user": ALICE}
    app.dependency_overrides[get_dao] = lambda: repo
    app.dependency_overrides[get_current_user] = lambda: state["user"]

    yield state

    app.dependency_overrides.pop(get_dao, None)
    app.dependency_overrides.pop(get_current_user, None)
    repo.clear()


def _create(**fields) -> dict:
    body = {"label": "Profil A", **fields}
    response = client.post("/profiles", json=body)
    assert response.status_code == 201, response.text
    return response.json()


class TestProfileEndpoints:
    """Tests for /profiles endpoints."""

    def test_list_empty(self):
        """Test listing when the user has no profiles."""
        response = client.get("/profiles")

        assert response.status_code == 200
        assert response.json() == []

    def test_create_sets_owner_from_user(self):
        """Test that the owner is the logged-in user."""
        created = _create(
            age_range="30-39", languages=[" norsk ", "engelsk"], notes="x"
        )

        assert created["owner_id"] == "alice"
        assert created["id"]
        assert created["age_range"] == "30-39"
        assert created["languages"] == ["norsk", "engelsk"]

    def test_create_rejects_owner_id_in_body(self):
        """Test that the client cannot choose the owner."""
        response = client.post("/profiles", json={"label": "A", "owner_id": "bob"})

        assert response.status_code == 422

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"label": "   "},
            {"label": "A", "age_range": "33"},
            {"label": "x" * 101},
            {"label": "A", "notes": "x" * 2001},
            {"label": "A", "languages": ["norsk"] * 21},
            {"label": "A", "experience_level": "lots"},
            {"label": "A", "practice_areas": ["x" * 101]},
            {"label": "A", "practice_areas": ["nervøsitet"] * 21},
        ],
    )
    def test_create_validation(self, body):
        """Test that invalid bodies are rejected."""
        assert client.post("/profiles", json=body).status_code == 422

    def test_list_only_own_profiles(self, overrides):
        """Test that a user only sees their own profiles."""
        _create(label="Alice's")
        overrides["user"] = BOB
        _create(label="Bob's")

        labels = [p["label"] for p in client.get("/profiles").json()]

        assert labels == ["Bob's"]

    def test_get_own_profile(self):
        """Test fetching one of the user's profiles."""
        created = _create()

        response = client.get(f"/profiles/{created['id']}")

        assert response.status_code == 200
        assert response.json() == created

    def test_get_missing_profile(self):
        """Test fetching a profile that does not exist."""
        assert client.get("/profiles/missing").status_code == 404

    def test_update_profile(self):
        """Test replacing the editable fields."""
        created = _create(notes="old")

        response = client.put(
            f"/profiles/{created['id']}",
            json={"label": "Edited", "age_range": "40-49"},
        )

        assert response.status_code == 200
        updated = response.json()
        assert updated["label"] == "Edited"
        assert updated["age_range"] == "40-49"
        assert updated["notes"] == ""
        assert updated["owner_id"] == "alice"
        assert updated["created_at"] == created["created_at"]

    def test_delete_profile(self):
        """Test deleting a profile."""
        created = _create()

        assert client.delete(f"/profiles/{created['id']}").status_code == 204
        assert client.get(f"/profiles/{created['id']}").status_code == 404
        assert client.delete(f"/profiles/{created['id']}").status_code == 404

    def test_other_users_profile_is_not_found(self, overrides):
        """Test that another user's profile looks like it does not exist."""
        created = _create()
        overrides["user"] = BOB
        url = f"/profiles/{created['id']}"

        assert client.get(url).status_code == 404
        assert client.put(url, json={"label": "Hacked"}).status_code == 404
        assert client.delete(url).status_code == 404

        overrides["user"] = ALICE
        assert client.get(url).json()["label"] == "Profil A"

    def test_experience_level_and_practice_areas(self):
        """Test that the new fields are stored, replaced and cleared."""
        created = _create(
            experience_level="little",
            practice_areas=[" Mestre nervøsitet ", "Fortelle om seg selv"],
        )
        assert created["experience_level"] == "little"
        assert created["practice_areas"] == [
            "Mestre nervøsitet",
            "Fortelle om seg selv",
        ]

        url = f"/profiles/{created['id']}"
        updated = client.put(
            url, json={"label": "Profil A", "experience_level": "extensive"}
        ).json()

        assert updated["experience_level"] == "extensive"
        assert updated["practice_areas"] == []
        assert client.get(url).json()["experience_level"] == "extensive"
