"""Authorization tests for chat and agent endpoints.

Uses the real AccessService and AuthService (see ``auth_env`` in conftest) so
the access-key and user checks are exercised end to end through FastAPI.
"""

from datetime import datetime, timedelta

import pytest

from src.models.accesskey import AccessKey
from tests.endpoints.conftest import INVALID_JWT, AuthEnv, sample_agent


ASK = "/api/chat/ask"


@pytest.fixture
def agent_id(auth_env: AuthEnv) -> str:
    return auth_env.agent_dao.add_agent(sample_agent()).id


@pytest.fixture
def valid_key(auth_env: AuthEnv, agent_id: str) -> str:
    return auth_env.access_service.generate_accesskey("valid", None, agent_id).key


def _ask_body(agent_id: str, access_key: str | None = None) -> dict:
    body = {"agent_id": agent_id, "chat_log": [], "include_progress": False}
    if access_key is not None:
        body["access_key"] = access_key
    return body


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestAskWithAccessKey:
    def test_valid_key_in_body_is_accepted(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(ASK, json=_ask_body(agent_id, valid_key))
        assert response.status_code == 200
        assert response.json() == {"response": f"answer from {agent_id}"}

    def test_valid_key_as_bearer_is_accepted(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(valid_key)
        )
        assert response.status_code == 200

    def test_wrong_key_is_rejected(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(ASK, json=_ask_body(agent_id, "wrong-key"))
        assert response.status_code == 403

    def test_wrong_key_as_bearer_is_rejected(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer("wrong-key")
        )
        assert response.status_code == 403

    def test_key_of_another_agent_is_rejected(self, auth_env, agent_id, valid_key):
        other_id = auth_env.agent_dao.add_agent(sample_agent("Other")).id
        response = auth_env.client.post(ASK, json=_ask_body(other_id, valid_key))
        assert response.status_code == 403

    def test_expired_key_is_rejected(self, auth_env, agent_id):
        agent = auth_env.agent_dao.get_agent_by_id(agent_id)
        agent.access_key.append(
            AccessKey(
                id="expired",
                key="expired-key-value",
                name="expired",
                expiry_date=datetime.now() - timedelta(days=1),
                created=datetime.now() - timedelta(days=2),
                last_use=None,
            )
        )
        auth_env.agent_dao.add_agent(agent)

        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id, "expired-key-value")
        )
        assert response.status_code == 403

    def test_no_credentials_is_unauthorized(self, auth_env, agent_id):
        response = auth_env.client.post(ASK, json=_ask_body(agent_id))
        assert response.status_code == 401

    def test_non_bearer_authorization_header_is_unauthorized(self, auth_env, agent_id):
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers={"Authorization": "Basic abc"}
        )
        assert response.status_code == 401


class TestAskWithUserToken:
    def test_owner_is_accepted(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(token)
        )
        assert response.status_code == 200

    def test_collaborator_is_accepted(self, auth_env, agent_id):
        _, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(token)
        )
        assert response.status_code == 200

    def test_user_without_access_is_rejected(self, auth_env, agent_id):
        _, token = auth_env.add_user("stranger")
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(token)
        )
        # require_user_has_agent_access hides agents the user cannot see.
        assert response.status_code == 404

    def test_invalid_jwt_is_unauthorized_not_an_access_key(self, auth_env, agent_id):
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(INVALID_JWT)
        )
        assert response.status_code == 401

    def test_invalid_jwt_cannot_pass_as_matching_access_key(self, auth_env, agent_id):
        # Even if a stored key happened to equal the token, a JWT-shaped bearer
        # must be validated as a JWT and never fall back to key matching.
        agent = auth_env.agent_dao.get_agent_by_id(agent_id)
        agent.access_key.append(
            AccessKey(
                id="jwtlike",
                key=INVALID_JWT,
                name="jwtlike",
                expiry_date=None,
                created=None,
                last_use=None,
            )
        )
        auth_env.agent_dao.add_agent(agent)
        response = auth_env.client.post(
            ASK, json=_ask_body(agent_id), headers=_bearer(INVALID_JWT)
        )
        assert response.status_code == 401


class TestGetAgentById:
    def test_wrong_key_is_rejected(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get(
            f"/agents/{agent_id}", headers=_bearer("wrong-key")
        )
        assert response.status_code == 403

    def test_valid_key_gets_scrubbed_agent(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get(
            f"/agents/{agent_id}", headers=_bearer(valid_key)
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["llm_api_key"] == ""
        assert data["embedding_api_key"] == ""
        assert data["access_key"] == []

    def test_owner_gets_full_agent(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        response = auth_env.client.get(f"/agents/{agent_id}", headers=_bearer(token))
        assert response.status_code == 200
        assert response.json()["llm_api_key"] == "sk-secret-llm-key"


class TestStaticAgentRoutesAreReachable:
    """Literal paths must not be swallowed by GET /agents/{agent_id}."""

    def test_agent_info_by_accesskey(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get(
            "/agents/agent-info-by-accesskey", headers=_bearer(valid_key)
        )
        assert response.status_code == 200
        assert response.json()["agent_id"] == agent_id

    def test_agent_info_by_accesskey_wrong_key(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get(
            "/agents/agent-info-by-accesskey", headers=_bearer("wrong-key")
        )
        assert response.status_code == 401

    def test_agent_info_by_accesskey_with_expiring_key(self, auth_env, agent_id):
        key = auth_env.access_service.generate_accesskey(
            "expiring", datetime.now() + timedelta(days=1), agent_id
        ).key
        response = auth_env.client.get(
            "/agents/agent-info-by-accesskey", headers=_bearer(key)
        )
        assert response.status_code == 200
        assert response.json()["agent_id"] == agent_id

    def test_agent_info(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get(
            "/agents/agent-info",
            params={"agent_id": agent_id},
            headers=_bearer(valid_key),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["llm_api_key"] == ""
        assert data["access_key"] == []

    def test_new_accesskey(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        response = auth_env.client.get(
            "/agents/new-accesskey",
            params={"agent_id": agent_id, "name": "fresh"},
            headers=_bearer(token),
        )
        assert response.status_code == 200
        assert response.json()["name"] == "fresh"

    @pytest.mark.parametrize(
        "path", ["/agents/chat-accesskey", "/agents/chat-access-key"]
    )
    def test_chat_accesskey(self, auth_env, agent_id, path):
        user, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        response = auth_env.client.get(
            path, params={"agent_id": agent_id}, headers=_bearer(token)
        )
        assert response.status_code == 200
        assert response.json()["name"] == f"Chat Access Key - {user.id}"

    def test_chat_accesskey_reuses_unexpired_key(self, auth_env, agent_id):
        _, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        first, second = (
            auth_env.client.get(
                "/agents/chat-accesskey",
                params={"agent_id": agent_id},
                headers=_bearer(token),
            )
            for _ in range(2)
        )
        assert first.status_code == second.status_code == 200
        assert first.json()["id"] == second.json()["id"]


class TestUserSearch:
    def test_excludes_requesting_user(self, auth_env):
        me, token = auth_env.add_user("me", name="Sam Searcher", email="me@test.com")
        other, _ = auth_env.add_user("other", name="Olive", email="olive@test.com")

        response = auth_env.client.get(
            "/agents/users/search", params={"q": "test.com"}, headers=_bearer(token)
        )

        assert response.status_code == 200
        ids = [result["id"] for result in response.json()]
        assert ids == [other.id]
        assert me.id not in ids
