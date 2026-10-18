"""Authorization tests for chat and agent endpoints.

Uses the real AccessService and AuthService (see ``auth_env`` in conftest) so
the access-key and user checks are exercised end to end through FastAPI.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.models.accesskey import AccessKey
from src.models.users.api_key import UserAPIKey
from src.utils.crypto_utils import encrypt_str
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

    def test_answer_is_wrapped_in_response_envelope(
        self, auth_env, agent_id, valid_key
    ):
        # Chat and XR clients read the answer at body["response"]["response"].
        response = auth_env.client.post(ASK, json=_ask_body(agent_id, valid_key))
        assert response.json() == {"response": {"response": f"answer from {agent_id}"}}

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
        auth_env.agent_dao.push_access_key(
            agent_id,
            AccessKey(
                id="expired",
                key="expired-key-value",
                name="expired",
                expiry_date=datetime.now(UTC) - timedelta(days=1),
                created=datetime.now(UTC) - timedelta(days=2),
                last_use=None,
            ),
        )

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
        auth_env.agent_dao.push_access_key(
            agent_id,
            AccessKey(
                id="jwtlike",
                key=INVALID_JWT,
                name="jwtlike",
                expiry_date=None,
                created=None,
                last_use=None,
            ),
        )
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
        assert "llm_api_key" not in data
        assert "embedding_api_key" not in data
        assert data["access_key"] == []

    @pytest.mark.parametrize("relation", ["owned_agents", "collaborating_agents"])
    def test_user_gets_agent_without_secrets(
        self, auth_env, agent_id, valid_key, relation
    ):
        _, token = auth_env.add_user("member", **{relation: [agent_id]})
        response = auth_env.client.get(f"/agents/{agent_id}", headers=_bearer(token))
        assert response.status_code == 200
        _assert_no_secrets(response.text, valid_key)
        data = response.json()
        assert "llm_api_key" not in data
        assert "embedding_api_key" not in data
        assert data["has_llm_api_key"] is True
        assert data["has_embedding_api_key"] is True
        assert data["llm_api_key_invalid"] is False
        [key] = data["access_key"]
        assert key["prefix"] == valid_key[:12]
        assert "key" not in key
        assert "key_hash" not in key


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
        assert "llm_api_key" not in data
        assert data["access_key"] == []
        _assert_no_secrets(response.text, valid_key)

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

    def test_chat_accesskey_mints_a_new_short_lived_key(self, auth_env, agent_id):
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
        assert first.json()["id"] != second.json()["id"]
        for response in (first, second):
            data = response.json()
            assert auth_env.access_service.authenticate(agent_id, data["key"])
            expiry = datetime.fromisoformat(data["expiry_date"])
            assert expiry.tzinfo is not None
            remaining = expiry - datetime.now(UTC)
            assert timedelta(hours=23) < remaining <= timedelta(hours=24)

    def test_chat_accesskey_prunes_expired_chat_keys(self, auth_env, agent_id):
        user, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        auth_env.agent_dao.push_access_key(
            agent_id,
            AccessKey(
                id="oldchat1",
                name=f"Chat Access Key - {user.id}",
                key_hash="0" * 64,
                expiry_date=datetime.now(UTC) - timedelta(hours=1),
            ),
        )

        response = auth_env.client.get(
            "/agents/chat-accesskey",
            params={"agent_id": agent_id},
            headers=_bearer(token),
        )
        assert response.status_code == 200
        ids = [k.id for k in auth_env.agent_dao.get_agent_by_id(agent_id).access_key]
        assert ids == [response.json()["id"]]


def _assert_no_secrets(text: str, *access_keys: str) -> None:
    """The response body must not contain any secret of the sample agent."""
    for secret in ("sk-secret-llm-key", "secret-embedding-key", *access_keys):
        assert secret not in text
    assert "key_hash" not in text
    assert "key_encrypted" not in text


class TestAccessKeySecrets:
    def _owner(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        return _bearer(token)

    def test_created_key_is_returned_once_and_stored_hashed(self, auth_env, agent_id):
        headers = self._owner(auth_env, agent_id)
        response = auth_env.client.post(
            f"/agents/{agent_id}/accesskeys/",
            params={"name": "xr headset"},
            headers=headers,
        )
        assert response.status_code == 200
        created = response.json()
        plaintext = created["key"]
        assert plaintext.startswith(f"rgd_{created['id']}_")
        assert created["prefix"] == plaintext[:12]
        assert "key_hash" not in created

        [stored] = auth_env.agent_dao.get_agent_by_id(agent_id).access_key
        assert stored.key is None
        assert plaintext not in stored.model_dump_json()

        for path in (
            "/agents",
            f"/agents/{agent_id}",
            f"/agents/{agent_id}/accesskeys",
        ):
            listed = auth_env.client.get(path, headers=headers)
            assert listed.status_code == 200, path
            _assert_no_secrets(listed.text, plaintext)

        assert auth_env.access_service.authenticate(agent_id, plaintext)

    def test_expiry_date_is_stored_as_utc(self, auth_env, agent_id):
        headers = self._owner(auth_env, agent_id)
        response = auth_env.client.post(
            f"/agents/{agent_id}/accesskeys/",
            params={"name": "tz", "expiry_date": "2999-01-01T12:00:00+02:00"},
            headers=headers,
        )
        assert response.status_code == 200
        [stored] = auth_env.agent_dao.get_agent_by_id(agent_id).access_key
        assert stored.expiry_date == datetime(2999, 1, 1, 10, 0, tzinfo=UTC)

    def test_access_key_list_has_metadata_only(self, auth_env, agent_id, valid_key):
        _, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        response = auth_env.client.get(
            f"/agents/{agent_id}/accesskeys", headers=_bearer(token)
        )
        assert response.status_code == 200
        _assert_no_secrets(response.text, valid_key)
        [key] = response.json()
        assert key["name"] == "valid"
        assert key["prefix"] == valid_key[:12]
        assert "key" not in key

    def test_legacy_plaintext_key_is_not_listed(self, auth_env, agent_id):
        auth_env.agent_dao.push_access_key(
            agent_id,
            AccessKey(id="legacy01", key="legacy-plaintext-value", name="legacy"),
        )
        headers = self._owner(auth_env, agent_id)

        for path in (
            "/agents",
            f"/agents/{agent_id}",
            f"/agents/{agent_id}/accesskeys",
        ):
            response = auth_env.client.get(path, headers=headers)
            assert response.status_code == 200, path
            assert "legacy-plaintext-value" not in response.text

    def test_revoke_access_key(self, auth_env, agent_id, valid_key):
        key_id = auth_env.agent_dao.get_agent_by_id(agent_id).access_key[0].id
        response = auth_env.client.delete(
            f"/agents/{agent_id}/accesskeys/{key_id}",
            headers=self._owner(auth_env, agent_id),
        )
        assert response.status_code == 200
        assert response.json() is True
        assert not auth_env.access_service.authenticate(agent_id, valid_key)

    def test_access_key_lists_external_info(self, auth_env, agent_id, valid_key):
        response = auth_env.client.get("/agents", headers=_bearer(valid_key))
        assert response.status_code == 200
        assert response.json() == [
            {"agent_id": agent_id, "name": "Test Agent", "roles": []}
        ]


class TestAgentWriteResponses:
    def _agent_body(self, **overrides) -> dict:
        body = {
            "name": "Written",
            "description": "d",
            "prompt": "p",
            "llm_model": "gpt-4",
            "llm_api_key": "sk-secret-llm-key",
            "embedding_model": "text-embedding-ada-002",
            "embedding_api_key": "secret-embedding-key",
            "last_updated": "2026-10-18T00:00:00Z",
        }
        body.update(overrides)
        return body

    def test_create_agent_returns_no_provider_keys(self, auth_env):
        _, token = auth_env.add_user("creator")
        response = auth_env.client.post(
            "/agents", json=self._agent_body(), headers=_bearer(token)
        )
        assert response.status_code == 200
        _assert_no_secrets(response.text)
        stored = auth_env.agent_dao.get_agent_by_id(response.json()["id"])
        assert stored.llm_api_key == "sk-secret-llm-key"

    def test_create_agent_ignores_access_keys_in_body(self, auth_env):
        _, token = auth_env.add_user("creator")
        body = self._agent_body(
            access_key=[{"id": "planted1", "name": "x", "key": "planted"}]
        )
        response = auth_env.client.post("/agents", json=body, headers=_bearer(token))
        assert response.status_code == 200
        stored = auth_env.agent_dao.get_agent_by_id(response.json()["id"])
        assert stored.access_key == []

    @pytest.mark.parametrize("relation", ["owned_agents", "collaborating_agents"])
    def test_update_with_empty_provider_keys_keeps_stored_keys(
        self, auth_env, agent_id, valid_key, relation
    ):
        _, token = auth_env.add_user("member", **{relation: [agent_id]})
        response = auth_env.client.patch(
            f"/agents/{agent_id}",
            json=self._agent_body(llm_api_key="", embedding_api_key=""),
            headers=_bearer(token),
        )
        assert response.status_code == 200
        _assert_no_secrets(response.text, valid_key)
        stored = auth_env.agent_dao.get_agent_by_id(agent_id)
        assert stored.name == "Written"
        assert stored.llm_api_key == "sk-secret-llm-key"
        assert stored.embedding_api_key == "secret-embedding-key"
        assert len(stored.access_key) == 1


class TestProviderApiKeyDetail:
    def test_detail_returns_metadata_without_the_key(self, auth_env):
        api_key = UserAPIKey(
            label="openai",
            provider="openai",
            usage="llm",
            key_encrypted=encrypt_str("sk-provider-secret-value"),
            redacted_key="sk-p****alue",
        )
        _, token = auth_env.add_user("keyholder", api_keys=[api_key])
        response = auth_env.client.get(
            f"/api-keys/{api_key.id}", headers=_bearer(token)
        )
        assert response.status_code == 200
        assert "sk-provider-secret-value" not in response.text
        data = response.json()
        assert "raw_key" not in data
        assert data["redacted_key"] == "sk-p****alue"
        assert data["id"] == api_key.id


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


class TestRevealAccessKey:
    def _create_reusable(self, auth_env, agent_id) -> tuple[dict, dict]:
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        headers = _bearer(token)
        response = auth_env.client.post(
            f"/agents/{agent_id}/accesskeys/",
            params={"name": "reusable", "view_once": "false"},
            headers=headers,
        )
        assert response.status_code == 200
        return response.json(), headers

    def _reveal(self, auth_env, agent_id, key_id, headers):
        return auth_env.client.get(
            f"/agents/{agent_id}/accesskeys/{key_id}/reveal", headers=headers
        )

    def test_create_reusable_key_stores_no_plaintext(self, auth_env, agent_id):
        created, headers = self._create_reusable(auth_env, agent_id)
        assert created["view_once"] is False
        assert created["revealable"] is True
        assert "key_encrypted" not in created

        [stored] = auth_env.agent_dao.get_agent_by_id(agent_id).access_key
        assert stored.key is None
        assert stored.key_hash
        assert stored.key_encrypted
        assert created["key"] not in stored.model_dump_json()

        for path in (
            "/agents",
            f"/agents/{agent_id}",
            f"/agents/{agent_id}/accesskeys",
        ):
            listed = auth_env.client.get(path, headers=headers)
            assert listed.status_code == 200, path
            _assert_no_secrets(listed.text, created["key"])
            assert stored.key_encrypted not in listed.text

        [listed_key] = auth_env.client.get(
            f"/agents/{agent_id}/accesskeys", headers=headers
        ).json()
        assert listed_key["view_once"] is False
        assert listed_key["revealable"] is True
        assert auth_env.access_service.authenticate(agent_id, created["key"])

    def test_view_once_is_the_default(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        response = auth_env.client.post(
            f"/agents/{agent_id}/accesskeys/",
            params={"name": "once"},
            headers=_bearer(token),
        )
        assert response.status_code == 200
        assert response.json()["view_once"] is True
        assert response.json()["revealable"] is False
        [stored] = auth_env.agent_dao.get_agent_by_id(agent_id).access_key
        assert stored.key_encrypted is None

    def test_owner_can_reveal(self, auth_env, agent_id):
        created, headers = self._create_reusable(auth_env, agent_id)
        response = self._reveal(auth_env, agent_id, created["id"], headers)
        assert response.status_code == 200
        assert response.json() == {"key": created["key"]}

    def test_collaborator_can_reveal(self, auth_env, agent_id):
        created, _ = self._create_reusable(auth_env, agent_id)
        _, token = auth_env.add_user("collab", collaborating_agents=[agent_id])
        response = self._reveal(auth_env, agent_id, created["id"], _bearer(token))
        assert response.status_code == 200
        assert response.json() == {"key": created["key"]}

    def test_access_key_caller_cannot_reveal(self, auth_env, agent_id):
        created, _ = self._create_reusable(auth_env, agent_id)
        response = self._reveal(
            auth_env, agent_id, created["id"], _bearer(created["key"])
        )
        assert response.status_code == 403
        assert created["key"] not in response.text

    def test_stranger_cannot_reveal(self, auth_env, agent_id):
        created, _ = self._create_reusable(auth_env, agent_id)
        _, token = auth_env.add_user("stranger")
        response = self._reveal(auth_env, agent_id, created["id"], _bearer(token))
        assert response.status_code == 404
        assert created["key"] not in response.text

    def test_unknown_key_is_not_found(self, auth_env, agent_id):
        _, headers = self._create_reusable(auth_env, agent_id)
        response = self._reveal(auth_env, agent_id, "missing0", headers)
        assert response.status_code == 404

    def test_view_once_key_is_conflict(self, auth_env, agent_id, valid_key):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        key_id = auth_env.agent_dao.get_agent_by_id(agent_id).access_key[0].id
        response = self._reveal(auth_env, agent_id, key_id, _bearer(token))
        assert response.status_code == 409
        assert "only once" in response.json()["detail"]
        assert valid_key not in response.text

    def test_undecryptable_key_is_conflict(self, auth_env, agent_id):
        from cryptography.fernet import Fernet

        foreign = Fernet(Fernet.generate_key()).encrypt(b"rgd_gone0001_x").decode()
        auth_env.agent_dao.push_access_key(
            agent_id,
            AccessKey(
                id="gone0001",
                name="rotated out",
                key_hash="0" * 64,
                key_encrypted=foreign,
                view_once=False,
            ),
        )
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        response = self._reveal(auth_env, agent_id, "gone0001", _bearer(token))
        assert response.status_code == 409
        assert "create a new one" in response.json()["detail"]
