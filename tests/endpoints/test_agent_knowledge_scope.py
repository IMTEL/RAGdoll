"""knowledge_scope round-trips through the agent create/update endpoints."""

import pytest

from tests.endpoints.conftest import AuthEnv, sample_agent


def _payload(**fields) -> dict:
    data = sample_agent().model_dump(exclude={"id", "access_key"})
    data.update(fields)
    return data


class TestEndpoints:
    @pytest.fixture
    def client(self, auth_env: AuthEnv):
        _, token = auth_env.add_user("knowledge-scope-user")
        auth_env.client.headers["Authorization"] = f"Bearer {token}"
        return auth_env.client

    def test_create_defaults_to_documents_first(self, client):
        r = client.post("/agents", json=_payload())
        assert r.status_code == 200, r.text
        assert r.json()["knowledge_scope"] == "documents_first"

    def test_create_and_update_round_trip(self, client, auth_env):
        r = client.post("/agents", json=_payload(knowledge_scope="documents_only"))
        assert r.status_code == 200, r.text
        agent_id = r.json()["id"]
        assert r.json()["knowledge_scope"] == "documents_only"
        assert auth_env.agent_dao.get_agent_by_id(agent_id).knowledge_scope == (
            "documents_only"
        )

        r = client.patch(
            f"/agents/{agent_id}", json=_payload(knowledge_scope="documents_first")
        )
        assert r.status_code == 200, r.text
        assert r.json()["knowledge_scope"] == "documents_first"
        assert client.get(f"/agents/{agent_id}").json()["knowledge_scope"] == (
            "documents_first"
        )

    def test_create_rejects_unknown_value(self, client):
        r = client.post("/agents", json=_payload(knowledge_scope="web"))
        assert r.status_code == 422
