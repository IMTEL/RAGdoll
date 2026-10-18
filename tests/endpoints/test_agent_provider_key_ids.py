"""Agents can reference a stored provider key (UserAPIKey) by id.

The backend resolves the reference against the agent owner's stored keys and
decrypts it server-side; the key itself is never returned.
"""

import pytest
from fastapi.testclient import TestClient

from src import pipeline
from src.main import app
from src.models.users.api_key import UserAPIKey
from src.routes import chat as chat_routes
from src.routes import upload
from src.utils.crypto_utils import encrypt_str
from src.utils.path_dependencies import get_auth
from src.utils.provider_keys import resolve_provider_keys
from tests.endpoints.conftest import AuthEnv, sample_agent


STORED_LLM_SECRET = "sk-stored-llm-secret"
STORED_EMB_SECRET = "stored-embedding-secret"


def _stored_key(secret: str, usage: str = "both", label: str = "k") -> UserAPIKey:
    return UserAPIKey(
        label=label,
        provider="openai",
        usage=usage,
        key_encrypted=encrypt_str(secret),
        redacted_key="****",
    )


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _agent_body(**overrides) -> dict:
    body = {
        "name": "Ref",
        "description": "d",
        "prompt": "p",
        "llm_model": "gpt-4",
        "llm_api_key": "",
        "embedding_model": "text-embedding-ada-002",
        "embedding_api_key": "",
        "last_updated": "2026-10-18T00:00:00Z",
    }
    body.update(overrides)
    return body


@pytest.fixture
def owner_env(auth_env: AuthEnv):
    """An owner with two stored keys and an agent that uses literal keys."""
    llm_key = _stored_key(STORED_LLM_SECRET, usage="llm")
    emb_key = _stored_key(STORED_EMB_SECRET, usage="embedding")
    agent_id = auth_env.agent_dao.add_agent(sample_agent()).id
    owner, token = auth_env.add_user(
        "owner", owned_agents=[agent_id], api_keys=[llm_key, emb_key]
    )
    return auth_env, owner, token, agent_id, llm_key, emb_key


def _owner_dao_user(auth_env: AuthEnv, user_id: str):
    return auth_env.user_dao.get_user_by_id(user_id)


class TestCreate:
    def test_create_with_stored_key_ids(self, auth_env):
        llm_key = _stored_key(STORED_LLM_SECRET, usage="llm")
        emb_key = _stored_key(STORED_EMB_SECRET, usage="both")
        _, token = auth_env.add_user("creator", api_keys=[llm_key, emb_key])

        response = auth_env.client.post(
            "/agents",
            json=_agent_body(
                llm_api_key_id=llm_key.id, embedding_api_key_id=emb_key.id
            ),
            headers=_bearer(token),
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["llm_api_key_id"] == llm_key.id
        assert data["embedding_api_key_id"] == emb_key.id
        assert data["has_llm_api_key"] is True
        assert data["has_embedding_api_key"] is True
        assert STORED_LLM_SECRET not in response.text
        stored = auth_env.agent_dao.get_agent_by_id(data["id"])
        assert stored.llm_api_key == ""
        assert stored.llm_api_key_id == llm_key.id

    def test_create_with_another_users_key_is_rejected(self, auth_env):
        foreign = _stored_key(STORED_LLM_SECRET)
        auth_env.add_user("someone-else", api_keys=[foreign])
        _, token = auth_env.add_user("creator")

        response = auth_env.client.post(
            "/agents",
            json=_agent_body(llm_api_key_id=foreign.id),
            headers=_bearer(token),
        )
        assert response.status_code == 404
        assert auth_env.agent_dao.get_agents() == []

    def test_create_with_key_for_the_wrong_usage_is_rejected(self, auth_env):
        emb_only = _stored_key(STORED_EMB_SECRET, usage="embedding")
        _, token = auth_env.add_user("creator", api_keys=[emb_only])

        response = auth_env.client.post(
            "/agents",
            json=_agent_body(llm_api_key_id=emb_only.id),
            headers=_bearer(token),
        )
        assert response.status_code == 400


class TestUpdate:
    def test_owner_sets_id_and_literal_key_is_cleared(self, owner_env):
        auth_env, _, token, agent_id, llm_key, _ = owner_env
        response = auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key_id=llm_key.id),
            headers=_bearer(token),
        )
        assert response.status_code == 200, response.text
        stored = auth_env.agent_dao.get_agent_by_id(agent_id)
        assert stored.llm_api_key_id == llm_key.id
        assert stored.llm_api_key == ""
        # The embedding key was not touched.
        assert stored.embedding_api_key == "secret-embedding-key"
        assert stored.embedding_api_key_id is None

    def test_explicit_literal_key_wins_and_clears_the_id(self, owner_env):
        auth_env, _, token, agent_id, llm_key, _ = owner_env
        auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key_id=llm_key.id),
            headers=_bearer(token),
        )

        response = auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key="sk-typed-key", llm_api_key_id=llm_key.id),
            headers=_bearer(token),
        )
        assert response.status_code == 200
        stored = auth_env.agent_dao.get_agent_by_id(agent_id)
        assert stored.llm_api_key == "sk-typed-key"
        assert stored.llm_api_key_id is None

    def test_empty_key_without_id_keeps_the_reference(self, owner_env):
        auth_env, _, token, agent_id, llm_key, _ = owner_env
        auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key_id=llm_key.id),
            headers=_bearer(token),
        )

        response = auth_env.client.patch(
            f"/agents/{agent_id}", json=_agent_body(), headers=_bearer(token)
        )
        assert response.status_code == 200
        stored = auth_env.agent_dao.get_agent_by_id(agent_id)
        assert stored.llm_api_key_id == llm_key.id

    def test_owner_cannot_use_a_foreign_key(self, owner_env):
        auth_env, _, token, agent_id, _, _ = owner_env
        foreign = _stored_key("sk-foreign")
        auth_env.add_user("stranger", api_keys=[foreign])

        response = auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key_id=foreign.id),
            headers=_bearer(token),
        )
        assert response.status_code == 404
        assert auth_env.agent_dao.get_agent_by_id(agent_id).llm_api_key_id is None

    def test_collaborator_cannot_change_the_reference(self, owner_env):
        auth_env, _, _, agent_id, llm_key, _ = owner_env
        own_key = _stored_key("sk-collab")
        _, collab_token = auth_env.add_user(
            "collab", collaborating_agents=[agent_id], api_keys=[own_key]
        )

        for key_id in (own_key.id, llm_key.id):
            response = auth_env.client.patch(
                f"/agents/{agent_id}",
                json=_agent_body(llm_api_key_id=key_id),
                headers=_bearer(collab_token),
            )
            assert response.status_code == 403
        assert auth_env.agent_dao.get_agent_by_id(agent_id).llm_api_key_id is None

    def test_collaborator_may_keep_the_existing_reference(self, owner_env):
        auth_env, _, token, agent_id, llm_key, _ = owner_env
        auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(llm_api_key_id=llm_key.id),
            headers=_bearer(token),
        )
        _, collab_token = auth_env.add_user("collab", collaborating_agents=[agent_id])

        response = auth_env.client.patch(
            f"/agents/{agent_id}",
            json=_agent_body(name="Edited", llm_api_key_id=llm_key.id),
            headers=_bearer(collab_token),
        )
        assert response.status_code == 200
        stored = auth_env.agent_dao.get_agent_by_id(agent_id)
        assert stored.name == "Edited"
        assert stored.llm_api_key_id == llm_key.id


def _reference_both(auth_env, token, agent_id, llm_key, emb_key):
    response = auth_env.client.patch(
        f"/agents/{agent_id}",
        json=_agent_body(llm_api_key_id=llm_key.id, embedding_api_key_id=emb_key.id),
        headers=_bearer(token),
    )
    assert response.status_code == 200, response.text


@pytest.fixture
def real_pipeline(monkeypatch):
    monkeypatch.setattr(
        chat_routes, "assemble_prompt_with_agent", pipeline.assemble_prompt_with_agent
    )


class TestResolution:
    def test_pipeline_uses_the_referenced_keys(
        self, owner_env, real_pipeline, monkeypatch
    ):
        auth_env, _, token, agent_id, llm_key, emb_key = owner_env
        _reference_both(auth_env, token, agent_id, llm_key, emb_key)
        seen = {}

        class _LLM:
            def generate(self, *args, **kwargs):
                return "answer"

        def fake_create_llm(*args, **kwargs):
            seen["llm"] = kwargs.get("api_key", args[2] if len(args) > 2 else None)
            return _LLM()

        def fake_retrieve(agent, query, documents):
            seen["embedding"] = agent.embedding_api_key
            return [], None

        monkeypatch.setattr(pipeline, "create_llm", fake_create_llm)
        monkeypatch.setattr(pipeline, "_retrieve_contexts", fake_retrieve)

        response = auth_env.client.post(
            "/api/chat/ask",
            json={
                "agent_id": agent_id,
                "chat_log": [{"role": "user", "content": "hi"}],
            },
            headers=_bearer(token),
        )
        assert response.status_code == 200, response.text
        assert seen == {"llm": STORED_LLM_SECRET, "embedding": STORED_EMB_SECRET}

    def test_deleted_stored_key_behaves_like_a_missing_key(
        self, owner_env, real_pipeline, monkeypatch
    ):
        auth_env, owner, token, agent_id, llm_key, emb_key = owner_env
        _reference_both(auth_env, token, agent_id, llm_key, emb_key)
        user = _owner_dao_user(auth_env, owner.id)
        user.remove_api_key(llm_key.id)
        auth_env.user_dao.set_user(user)
        monkeypatch.setattr(pipeline, "_retrieve_contexts", lambda *a: ([], None))

        agent = resolve_provider_keys(
            auth_env.agent_dao.get_agent_by_id(agent_id), auth_env.user_dao
        )
        assert agent.llm_api_key == ""
        assert agent.llm_api_key_invalid is True
        assert agent.embedding_api_key == STORED_EMB_SECRET

        listed = auth_env.client.get(f"/agents/{agent_id}", headers=_bearer(token))
        assert listed.status_code == 200
        assert listed.json()["llm_api_key_invalid"] is True
        assert listed.json()["embedding_api_key_invalid"] is False

        # The real Idun client refuses an empty key: 401, not a 500.
        response = auth_env.client.post(
            "/api/chat/ask",
            json={
                "agent_id": agent_id,
                "chat_log": [{"role": "user", "content": "hi"}],
            },
            headers=_bearer(token),
        )
        assert response.status_code == 401

    def test_upload_uses_the_referenced_embedding_key(
        self, owner_env, monkeypatch, tmp_path
    ):
        auth_env, _, token, agent_id, llm_key, emb_key = owner_env
        _reference_both(auth_env, token, agent_id, llm_key, emb_key)
        seen = []
        monkeypatch.setattr(upload, "agent_dao", auth_env.agent_dao)
        monkeypatch.setattr(upload, "TEMP_FILES_DIR", tmp_path / "temp_files")
        monkeypatch.setattr(
            upload,
            "_process_document_background",
            lambda **kw: seen.append(kw["embedding_api_key"]),
        )
        app.dependency_overrides[get_auth] = lambda: None
        try:
            response = TestClient(app).post(
                "/upload/agent",
                params={"agent_id": agent_id},
                files={"file": ("x.txt", b"hello")},
            )
        finally:
            app.dependency_overrides.pop(get_auth, None)

        assert response.status_code == 200, response.text
        assert seen == [STORED_EMB_SECRET]
