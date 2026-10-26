"""Tests for the message feedback endpoints.

Uses the real AuthService (see ``auth_env`` in conftest), so the owner,
collaborator and stranger checks run through FastAPI. The feedback DAO is
replaced by the in-memory mock.
"""

import pytest

from src.main import app
from src.routes import agents as agents_routes
from src.routes.feedback import get_dao
from tests.endpoints.conftest import AuthEnv, sample_agent
from tests.mocks.mock_feedback_dao import MockFeedbackDAO


@pytest.fixture
def repo(auth_env: AuthEnv, monkeypatch):
    repo = MockFeedbackDAO()
    repo.clear()
    app.dependency_overrides[get_dao] = lambda: repo
    # Deleting an agent also deletes its feedback
    monkeypatch.setattr(agents_routes, "feedback_dao", repo)

    yield repo

    app.dependency_overrides.pop(get_dao, None)
    repo.clear()


@pytest.fixture
def agent_id(auth_env: AuthEnv) -> str:
    return auth_env.agent_dao.add_agent(sample_agent()).id


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _owner(auth_env: AuthEnv, agent_id: str) -> dict:
    _, token = auth_env.add_user("owner", owned_agents=[agent_id])
    return _bearer(token)


def _url(agent_id: str) -> str:
    return f"/agents/{agent_id}/feedback"


def _body(**fields) -> dict:
    return {"rating": "bad", "agent_message": "Hva er din største svakhet?", **fields}


class TestCreateFeedback:
    def test_owner_can_rate(self, auth_env, repo, agent_id):
        response = auth_env.client.post(
            _url(agent_id),
            json=_body(comment="  For direkte  ", user_message="Hei"),
            headers=_owner(auth_env, agent_id),
        )

        assert response.status_code == 201, response.text
        created = response.json()
        assert created["id"]
        assert created["agent_id"] == agent_id
        assert created["rating"] == "bad"
        assert created["comment"] == "For direkte"
        assert created["user_message"] == "Hei"
        assert repo.get_by_id(created["id"]) is not None

    def test_author_is_the_logged_in_user(self, auth_env, repo, agent_id):
        user, token = auth_env.add_user("owner", owned_agents=[agent_id])

        response = auth_env.client.post(
            _url(agent_id), json=_body(), headers=_bearer(token)
        )

        assert response.json()["author_id"] == user.id

    def test_collaborator_can_rate(self, auth_env, repo, agent_id):
        _, token = auth_env.add_user("collab", collaborating_agents=[agent_id])

        response = auth_env.client.post(
            _url(agent_id), json=_body(), headers=_bearer(token)
        )

        assert response.status_code == 201

    def test_stranger_gets_404(self, auth_env, repo, agent_id):
        _, token = auth_env.add_user("stranger")

        response = auth_env.client.post(
            _url(agent_id), json=_body(), headers=_bearer(token)
        )

        assert response.status_code == 404
        assert repo.get_recent_by_agent(agent_id, 10) == []

    def test_requires_login(self, auth_env, repo, agent_id):
        response = auth_env.client.post(_url(agent_id), json=_body())

        assert response.status_code in (401, 403)

    def test_access_key_cannot_rate(self, auth_env, repo, agent_id):
        key = auth_env.access_service.generate_accesskey("vr", None, agent_id).key

        response = auth_env.client.post(
            _url(agent_id), json=_body(), headers=_bearer(key)
        )

        assert response.status_code == 403

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"rating": "ok", "agent_message": "x"},
            {"rating": "good"},
            {"rating": "good", "agent_message": "   "},
            {"rating": "good", "agent_message": "x" * 4001},
            {"rating": "good", "agent_message": "x", "comment": "x" * 501},
            {"rating": "good", "agent_message": "x", "author_id": "bob"},
            {"rating": "good", "agent_message": "x", "agent_id": "other"},
        ],
    )
    def test_invalid_body_is_rejected(self, auth_env, repo, agent_id, body):
        response = auth_env.client.post(
            _url(agent_id), json=body, headers=_owner(auth_env, agent_id)
        )

        assert response.status_code == 422


class TestListFeedback:
    def test_lists_only_the_agent(self, auth_env, repo, agent_id):
        other_id = auth_env.agent_dao.add_agent(sample_agent("Other")).id
        _, token = auth_env.add_user("owner", owned_agents=[agent_id, other_id])
        headers = _bearer(token)
        auth_env.client.post(_url(agent_id), json=_body(), headers=headers)
        auth_env.client.post(_url(other_id), json=_body(), headers=headers)

        response = auth_env.client.get(_url(agent_id), headers=headers)

        assert response.status_code == 200
        assert [f["agent_id"] for f in response.json()] == [agent_id]

    def test_limit_is_bounded(self, auth_env, repo, agent_id):
        headers = _owner(auth_env, agent_id)

        assert (
            auth_env.client.get(
                _url(agent_id), params={"limit": 0}, headers=headers
            ).status_code
            == 422
        )
        assert (
            auth_env.client.get(
                _url(agent_id), params={"limit": 201}, headers=headers
            ).status_code
            == 422
        )

    def test_stranger_gets_404(self, auth_env, repo, agent_id):
        _, token = auth_env.add_user("stranger")

        response = auth_env.client.get(_url(agent_id), headers=_bearer(token))

        assert response.status_code == 404


class TestDeleteFeedback:
    def test_owner_can_delete(self, auth_env, repo, agent_id):
        headers = _owner(auth_env, agent_id)
        created = auth_env.client.post(
            _url(agent_id), json=_body(), headers=headers
        ).json()

        response = auth_env.client.delete(
            f"{_url(agent_id)}/{created['id']}", headers=headers
        )

        assert response.status_code == 204
        assert repo.get_by_id(created["id"]) is None

    def test_unknown_feedback_is_404(self, auth_env, repo, agent_id):
        response = auth_env.client.delete(
            f"{_url(agent_id)}/missing", headers=_owner(auth_env, agent_id)
        )

        assert response.status_code == 404

    def test_feedback_of_another_agent_is_404(self, auth_env, repo, agent_id):
        other_id = auth_env.agent_dao.add_agent(sample_agent("Other")).id
        _, token = auth_env.add_user("owner", owned_agents=[agent_id, other_id])
        headers = _bearer(token)
        created = auth_env.client.post(
            _url(other_id), json=_body(), headers=headers
        ).json()

        response = auth_env.client.delete(
            f"{_url(agent_id)}/{created['id']}", headers=headers
        )

        assert response.status_code == 404
        assert repo.get_by_id(created["id"]) is not None

    def test_stranger_cannot_delete(self, auth_env, repo, agent_id):
        created = auth_env.client.post(
            _url(agent_id), json=_body(), headers=_owner(auth_env, agent_id)
        ).json()
        _, token = auth_env.add_user("stranger")

        response = auth_env.client.delete(
            f"{_url(agent_id)}/{created['id']}", headers=_bearer(token)
        )

        assert response.status_code == 404
        assert repo.get_by_id(created["id"]) is not None


class TestDeleteAgentDeletesFeedback:
    def test_feedback_is_deleted_with_the_agent(self, auth_env, repo, agent_id):
        headers = _owner(auth_env, agent_id)
        auth_env.client.post(_url(agent_id), json=_body(), headers=headers)

        response = auth_env.client.delete(f"/agents/{agent_id}", headers=headers)

        assert response.status_code == 200, response.text
        assert repo.get_recent_by_agent(agent_id, 10) == []
