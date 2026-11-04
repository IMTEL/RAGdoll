"""Publishing agents to The Candidate, with the real authorization wiring."""

import pytest

from src.config import Config
from src.models.agent import (
    PUBLIC_DESCRIPTION_MAX_LENGTH,
    PUBLIC_NAME_MAX_LENGTH,
    Role,
)
from src.routes import chat as chat_routes
from tests.endpoints.conftest import AuthEnv, sample_agent


CLIENT_KEY = "candidate-test-key"


@pytest.fixture
def env(auth_env: AuthEnv, monkeypatch):
    monkeypatch.setattr(Config(), "CANDIDATE_CLIENT_KEY", CLIENT_KEY)
    return auth_env


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def make_agent(env: AuthEnv, name: str = "Intern: intervjuer for Ola"):
    agent = sample_agent(name)
    agent.description = '{"type": "interviewer", "answers": {"profil": "secret"}}'
    agent.prompt = "You are a strict interviewer."
    agent.roles = [Role(name="Intern rolle", description="Persona text")]
    return env.agent_dao.add_agent(agent)


def owned_agent(env: AuthEnv, name: str = "Intern: intervjuer for Ola"):
    agent = make_agent(env, name)
    _, token = env.add_user("owner", owned_agents=[agent.id])
    return agent, token


def publish(env: AuthEnv, agent_id: str, token: str, **body):
    payload = {"public_name": "Kari fra NAV", "public_description": "Øv på intervju"}
    payload.update(body)
    return env.client.post(
        f"/agents/{agent_id}/publish", json=payload, headers=bearer(token)
    )


def ask(env: AuthEnv, agent_id: str, key: str):
    return env.client.post(
        "/api/chat/ask",
        json={"agent_id": agent_id, "chat_log": []},
        headers=bearer(key),
    )


class TestPublish:
    def test_owner_publishes(self, env):
        agent, token = owned_agent(env)

        response = publish(env, agent.id, token)

        assert response.status_code == 200
        data = response.json()
        assert data["is_published"] is True
        assert data["public_name"] == "Kari fra NAV"
        assert data["public_description"] == "Øv på intervju"
        assert data["published_at"]
        assert env.agent_dao.get_agent_by_id(agent.id).is_published is True

    def test_collaborator_cannot_publish(self, env):
        agent = make_agent(env)
        _, token = env.add_user("collaborator", collaborating_agents=[agent.id])

        assert publish(env, agent.id, token).status_code == 404
        assert env.agent_dao.get_agent_by_id(agent.id).is_published is False

    def test_anonymous_cannot_publish(self, env):
        agent = make_agent(env)

        response = env.client.post(
            f"/agents/{agent.id}/publish", json={"public_name": "Kari"}
        )

        assert response.status_code in (401, 403)

    @pytest.mark.parametrize(
        "body",
        [
            {"public_name": "   "},
            {"public_name": "x" * (PUBLIC_NAME_MAX_LENGTH + 1)},
            {"public_description": "x" * (PUBLIC_DESCRIPTION_MAX_LENGTH + 1)},
            {"extra": 1},
        ],
    )
    def test_rejects_invalid_body(self, env, body):
        agent, token = owned_agent(env)

        assert publish(env, agent.id, token, **body).status_code == 422

    def test_unpublish_hides_but_keeps_text(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        response = env.client.post(
            f"/agents/{agent.id}/unpublish", headers=bearer(token)
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_published"] is False
        assert data["published_at"] is None
        assert data["public_name"] == "Kari fra NAV"

    def test_owner_sees_publish_fields_in_agent_list(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        [listed] = env.client.get("/agents", headers=bearer(token)).json()

        assert listed["is_published"] is True
        assert listed["public_name"] == "Kari fra NAV"

    def test_edit_keeps_published_state(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)
        edited = sample_agent("Nytt navn").model_dump()

        response = env.client.post(
            f"/agents/{agent.id}", json=edited, headers=bearer(token)
        )

        assert response.status_code == 200
        stored = env.agent_dao.get_agent_by_id(agent.id)
        assert stored.name == "Nytt navn"
        assert stored.is_published is True
        assert stored.public_name == "Kari fra NAV"

    def test_edit_cannot_publish(self, env):
        agent, token = owned_agent(env)
        edited = sample_agent().model_dump() | {"is_published": True}

        env.client.post(f"/agents/{agent.id}", json=edited, headers=bearer(token))

        assert env.agent_dao.get_agent_by_id(agent.id).is_published is False

    def test_create_cannot_publish(self, env):
        _, token = env.add_user("creator")
        body = sample_agent().model_dump() | {
            "is_published": True,
            "public_name": "Snarvei",
        }

        response = env.client.post("/agents", json=body, headers=bearer(token))

        assert response.status_code == 200
        assert response.json()["is_published"] is False
        assert (
            env.client.get("/agents/published", headers=bearer(CLIENT_KEY)).json() == []
        )


class TestPublishedList:
    def test_lists_only_published_agents(self, env):
        agent, token = owned_agent(env)
        make_agent(env, "Privat")
        publish(env, agent.id, token)

        response = env.client.get("/agents/published", headers=bearer(CLIENT_KEY))

        assert response.status_code == 200
        assert response.json() == [
            {
                "agent_id": agent.id,
                "name": "Kari fra NAV",
                "description": "Øv på intervju",
            }
        ]

    def test_hides_prompt_profile_and_keys(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        body = env.client.get("/agents/published", headers=bearer(CLIENT_KEY)).text

        for secret in (
            "strict interviewer",
            "profil",
            "Persona text",
            "Intern",
            "sk-secret",
        ):
            assert secret not in body

    def test_rejects_missing_or_wrong_key(self, env):
        assert env.client.get("/agents/published").status_code in (401, 403)
        response = env.client.get("/agents/published", headers=bearer("wrong"))
        assert response.status_code == 403

    def test_rejects_a_user_token(self, env):
        _, token = env.add_user("someone")

        response = env.client.get("/agents/published", headers=bearer(token))

        assert response.status_code == 403

    def test_off_when_no_key_is_configured(self, env, monkeypatch):
        monkeypatch.setattr(Config(), "CANDIDATE_CLIENT_KEY", "")

        response = env.client.get("/agents/published", headers=bearer("anything"))

        assert response.status_code == 403


class TestCandidateChat:
    def test_client_key_chats_with_published_agent(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        response = ask(env, agent.id, CLIENT_KEY)

        assert response.status_code == 200
        assert response.json()["response"]["response"] == f"answer from {agent.id}"

    def test_client_key_in_body_also_works(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        response = env.client.post(
            "/api/chat/ask",
            json={"agent_id": agent.id, "access_key": CLIENT_KEY, "chat_log": []},
        )

        assert response.status_code == 200

    def test_client_key_rejected_for_unpublished_agent(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)
        env.client.post(f"/agents/{agent.id}/unpublish", headers=bearer(token))

        assert ask(env, agent.id, CLIENT_KEY).status_code == 403

    def test_client_key_does_not_unlock_agent_configuration(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        for path in (f"/agents/{agent.id}", f"/agents/agent-info?agent_id={agent.id}"):
            response = env.client.get(path, headers=bearer(CLIENT_KEY))
            assert response.status_code in (401, 403), path

    def test_client_key_does_not_list_agents(self, env):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)

        response = env.client.get("/agents", headers=bearer(CLIENT_KEY))

        assert response.json() == []

    def test_client_key_chats_as_the_first_role(self, env, monkeypatch):
        agent, token = owned_agent(env)
        publish(env, agent.id, token)
        used_roles = []
        monkeypatch.setattr(
            chat_routes,
            "assemble_prompt_with_agent",
            lambda command, agent: (
                used_roles.append(command.active_role_id) or {"response": "ok"}
            ),
        )

        assert ask(env, agent.id, CLIENT_KEY).status_code == 200
        assert used_roles == ["Intern rolle"]


class TestCandidateObserver:
    """The observer gives feedback after an interview in The Candidate."""

    def test_client_key_chats_with_unpublished_observer(self, env, monkeypatch):
        observer = make_agent(env, "Observatør")
        monkeypatch.setattr(Config(), "CANDIDATE_OBSERVER_AGENT_ID", observer.id)

        response = ask(env, observer.id, CLIENT_KEY)

        assert response.status_code == 200
        assert response.json()["response"]["response"] == f"answer from {observer.id}"

    def test_no_observer_when_not_configured(self, env, monkeypatch):
        observer = make_agent(env, "Observatør")
        monkeypatch.setattr(Config(), "CANDIDATE_OBSERVER_AGENT_ID", "")

        assert ask(env, observer.id, CLIENT_KEY).status_code == 403

    def test_other_unpublished_agents_stay_closed(self, env, monkeypatch):
        observer = make_agent(env, "Observatør")
        other = make_agent(env, "Privat")
        monkeypatch.setattr(Config(), "CANDIDATE_OBSERVER_AGENT_ID", observer.id)

        assert ask(env, other.id, CLIENT_KEY).status_code == 403
