"""Every route must reject anonymous requests unless explicitly allowlisted.

Also covers ``require_user_or_access_key`` (user JWT, or access key plus
``X-Agent-Id``), the ``PUBLIC_SPEECH_ENDPOINTS`` opt-out, the scoped debug
log route and the upload status route.
"""

import json
import logging
import re

import pytest
from fastapi.routing import APIRoute, iter_route_contexts

from src.config import Config
from src.main import app, warn_if_speech_endpoints_public
from src.routes import chat as chat_routes
from src.routes import translation as translation_routes
from src.routes import transliteration as transliteration_routes
from src.utils.global_logs import failure_log, progress_log
from tests.endpoints.conftest import INVALID_JWT, AuthEnv, sample_agent


PUBLIC_ROUTES = {
    "/",
    "/ping",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
}

# The progress routes authenticate with their own agent_id + access_key
# (an `access_key` header, or `access_key` in the JSON body), checked inside
# the handler. They are covered by their own tests, not by this guard.
OWN_AUTH_PREFIXES = ("/api/progress",)

# Routes where an empty request gets 422 from FastAPI's own validation: the
# credentials (or the agent_id they are checked against) live in a required
# request field, so the guard's body-less request never reaches the handler.
# The chat routes accept the access key inside the Command body
# (``access_key``) or the ``data`` form field, as well as a bearer header.
# With a valid body the handlers authenticate first, before any LLM,
# transcription or TTS work, and answer 401/403 without credentials:
# TestChatRoutesAuthenticateBeforeWork below covers that.
VALIDATION_FIRST_ROUTES = {
    ("POST", "/api/chat/ask"),
    ("POST", "/api/chat/askWithSpeech"),
    ("POST", "/api/chat/askTranscribe"),
    ("POST", "/api/chat/askTranscribeWithSpeech"),
}

DUMMY_PATH_VALUE = "000000000000000000000000"


def _api_routes() -> list[tuple[str, str]]:
    """Every (method, full path) served by the app.

    ``app.routes`` holds included routers as opaque entries in this FastAPI
    version, so walk them with ``iter_route_contexts`` to get the APIRoutes
    with their prefixed paths.
    """
    return [
        (method, ctx.path)
        for ctx in iter_route_contexts(app.routes)
        if isinstance(ctx.original_route, APIRoute)
        for method in sorted(ctx.methods)
    ]


def _guarded_routes() -> list[tuple[str, str]]:
    return [
        (method, path)
        for method, path in _api_routes()
        if path not in PUBLIC_ROUTES and not path.startswith(OWN_AUTH_PREFIXES)
    ]


def test_route_walk_sees_included_routers():
    assert ("POST", "/api/translate") in _api_routes()
    assert len(_guarded_routes()) > 40


def _fail(*_args, **_kwargs):
    raise AssertionError("handler ran without credentials")


@pytest.fixture
def no_heavy_handlers(monkeypatch):
    """Stop any handler that slips past auth from loading models or the network."""
    for name in (
        "warmup_whisper_model",
        "get_tts_service",
        "transcribe_audio",
        "assemble_prompt_with_agent",
    ):
        monkeypatch.setattr(chat_routes, name, _fail)
    monkeypatch.setattr(translation_routes, "get_translation_service", _fail)
    monkeypatch.setattr(transliteration_routes, "get_transliteration_service", _fail)


@pytest.fixture
def speech_endpoints_private(monkeypatch):
    monkeypatch.setattr(Config(), "PUBLIC_SPEECH_ENDPOINTS", False)


@pytest.mark.parametrize(("method", "path"), _guarded_routes())
def test_route_rejects_anonymous_request(
    auth_env: AuthEnv,
    speech_endpoints_private,
    no_heavy_handlers,
    method: str,
    path: str,
):
    url = re.sub(r"\{[^}]+\}", DUMMY_PATH_VALUE, path)
    response = auth_env.client.request(method, url)

    allowed = {401, 403}
    if (method, path) in VALIDATION_FIRST_ROUTES:
        allowed.add(422)
    assert response.status_code in allowed, (
        f"{method} {path} answered {response.status_code} without credentials"
    )


class _FakeTranslation:
    def to_dict(self) -> dict:
        return {"translatedText": "hola"}


class _FakeTranslationService:
    def translate(self, text, source, target):
        return _FakeTranslation()


@pytest.fixture
def fake_translation(monkeypatch):
    monkeypatch.setattr(
        translation_routes,
        "get_translation_service",
        lambda: _FakeTranslationService(),
    )


@pytest.fixture
def agent_id(auth_env: AuthEnv) -> str:
    return auth_env.agent_dao.add_agent(sample_agent()).id


@pytest.fixture
def valid_key(auth_env: AuthEnv, agent_id: str) -> str:
    return auth_env.access_service.generate_accesskey("valid", None, agent_id).key


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


TRANSLATE = "/api/translate"
TRANSLATE_BODY = {"text": "hello", "source": "en", "target": "es"}


@pytest.mark.usefixtures("speech_endpoints_private", "fake_translation")
class TestRequireUserOrAccessKey:
    def test_no_credentials_is_401(self, auth_env):
        response = auth_env.client.post(TRANSLATE, json=TRANSLATE_BODY)
        assert response.status_code == 401

    def test_valid_user_is_accepted(self, auth_env):
        _, token = auth_env.add_user("speaker")
        response = auth_env.client.post(
            TRANSLATE, json=TRANSLATE_BODY, headers=_bearer(token)
        )
        assert response.status_code == 200
        assert response.json() == {"translatedText": "hola"}

    def test_invalid_jwt_is_401(self, auth_env):
        response = auth_env.client.post(
            TRANSLATE, json=TRANSLATE_BODY, headers=_bearer(INVALID_JWT)
        )
        assert response.status_code == 401

    def test_valid_access_key_with_agent_header_is_accepted(
        self, auth_env, agent_id, valid_key
    ):
        response = auth_env.client.post(
            TRANSLATE,
            json=TRANSLATE_BODY,
            headers={**_bearer(valid_key), "X-Agent-Id": agent_id},
        )
        assert response.status_code == 200

    def test_wrong_access_key_is_403(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(
            TRANSLATE,
            json=TRANSLATE_BODY,
            headers={**_bearer("wrong-key"), "X-Agent-Id": agent_id},
        )
        assert response.status_code == 403

    def test_access_key_for_another_agent_is_403(self, auth_env, agent_id, valid_key):
        other_id = auth_env.agent_dao.add_agent(sample_agent("Other")).id
        response = auth_env.client.post(
            TRANSLATE,
            json=TRANSLATE_BODY,
            headers={**_bearer(valid_key), "X-Agent-Id": other_id},
        )
        assert response.status_code == 403

    def test_access_key_without_agent_header_is_401(self, auth_env, valid_key):
        response = auth_env.client.post(
            TRANSLATE, json=TRANSLATE_BODY, headers=_bearer(valid_key)
        )
        assert response.status_code == 401


SPEECH_ROUTES = [
    ("POST", "/api/chat/transcribe"),
    ("POST", "/api/chat/tts"),
    ("POST", "/api/chat/tts/warmup"),
    ("POST", "/api/chat/stt/warmup"),
    ("POST", "/api/translate"),
    ("POST", "/api/translate/batch"),
    ("POST", "/api/translate/warmup"),
    ("POST", "/api/transliterate"),
]


class TestPublicSpeechEndpointsFlag:
    def test_flag_opens_only_speech_and_translation_routes(
        self, auth_env, monkeypatch, no_heavy_handlers
    ):
        monkeypatch.setattr(Config(), "PUBLIC_SPEECH_ENDPOINTS", True)
        monkeypatch.setattr(
            translation_routes,
            "get_translation_service",
            lambda: _FakeTranslationService(),
        )

        response = auth_env.client.post(TRANSLATE, json=TRANSLATE_BODY)
        assert response.status_code == 200

        speech_paths = {path for _, path in SPEECH_ROUTES}
        for method, path in _guarded_routes():
            url = re.sub(r"\{[^}]+\}", DUMMY_PATH_VALUE, path)
            if path in speech_paths:
                continue
            response = auth_env.client.request(method, url)
            allowed = {401, 403}
            if (method, path) in VALIDATION_FIRST_ROUTES:
                allowed.add(422)
            assert response.status_code in allowed, (
                f"{method} {path} opened by PUBLIC_SPEECH_ENDPOINTS"
            )
        for path in ("/api/debug/logs", f"/upload/status/{DUMMY_PATH_VALUE}"):
            assert auth_env.client.get(path).status_code == 401

    def test_flag_covers_every_speech_route(self):
        assert set(SPEECH_ROUTES) <= set(_api_routes())

    def test_startup_warning_when_public(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.main"):
            warn_if_speech_endpoints_public(True)
        assert "PUBLIC_SPEECH_ENDPOINTS" in caplog.text

    def test_no_warning_when_private(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.main"):
            warn_if_speech_endpoints_public(False)
        assert caplog.text == ""


@pytest.fixture
def clean_logs():
    saved_progress, saved_failure = list(progress_log), list(failure_log)
    progress_log.clear()
    failure_log.clear()
    yield
    progress_log[:] = saved_progress
    failure_log[:] = saved_failure


@pytest.mark.usefixtures("clean_logs")
class TestDebugLogs:
    def test_returns_only_entries_for_users_agents(self, auth_env):
        _, token = auth_env.add_user(
            "dev", owned_agents=["owned"], collaborating_agents=["shared"]
        )
        progress_log.extend(
            [
                {"agent_id": "owned", "task_name": "a"},
                {"agent_id": "shared", "task_name": "b"},
                {"agent_id": "foreign", "task_name": "c"},
                {"task_name": "no-agent"},
            ]
        )
        failure_log.extend(
            [{"agent_id": "foreign", "error": "x"}, {"agent_id": "owned", "error": "y"}]
        )

        response = auth_env.client.get("/api/debug/logs", headers=_bearer(token))

        assert response.status_code == 200
        body = response.json()
        assert [entry["task_name"] for entry in body["progressLog"]] == ["a", "b"]
        assert body["failureLog"] == [{"agent_id": "owned", "error": "y"}]

    def test_access_key_is_rejected(self, auth_env, valid_key):
        response = auth_env.client.get("/api/debug/logs", headers=_bearer(valid_key))
        assert response.status_code == 403


class TestBootstrapRoute:
    def test_requires_user_even_when_enabled(self, auth_env, monkeypatch):
        monkeypatch.setenv("AUTH_BOOTSTRAP_ENABLED", "true")
        response = auth_env.client.post(
            "/api/debug/bootstrap-keycloak-user", json={"provider_user_id": "x"}
        )
        assert response.status_code == 401

    def test_still_disabled_by_default_for_users(self, auth_env, monkeypatch):
        monkeypatch.delenv("AUTH_BOOTSTRAP_ENABLED", raising=False)
        _, token = auth_env.add_user("dev")
        response = auth_env.client.post(
            "/api/debug/bootstrap-keycloak-user",
            json={"provider_user_id": "x"},
            headers=_bearer(token),
        )
        assert response.status_code == 404


@pytest.mark.usefixtures("clean_logs")
class TestUploadStatus:
    def _add_task(self, agent_id: str) -> str:
        progress_log.append(
            {"task_id": "task-1", "agent_id": agent_id, "status": "processing"}
        )
        return "/upload/status/task-1"

    def test_owner_sees_status(self, auth_env, agent_id):
        _, token = auth_env.add_user("owner", owned_agents=[agent_id])
        url = self._add_task(agent_id)
        response = auth_env.client.get(url, headers=_bearer(token))
        assert response.status_code == 200
        assert response.json()["task_id"] == "task-1"
        assert response.json()["status"] == "processing"

    def test_access_key_for_agent_sees_status(self, auth_env, agent_id, valid_key):
        url = self._add_task(agent_id)
        response = auth_env.client.get(
            url, headers={**_bearer(valid_key), "X-Agent-Id": agent_id}
        )
        assert response.status_code == 200

    def test_other_user_gets_404(self, auth_env, agent_id):
        _, token = auth_env.add_user("stranger")
        url = self._add_task(agent_id)
        response = auth_env.client.get(url, headers=_bearer(token))
        assert response.status_code == 404


def _command(agent_id: str, access_key: str | None = None) -> dict:
    body = {"agent_id": agent_id, "chat_log": [], "include_progress": False}
    if access_key is not None:
        body["access_key"] = access_key
    return body


AUDIO = {"audio": ("q.wav", b"RIFF0000WAVE", "audio/wav")}
TRANSCRIBE_ROUTES = [
    "/api/chat/askTranscribe",
    "/api/chat/askTranscribeWithSpeech",
]


@pytest.mark.usefixtures("speech_endpoints_private", "no_heavy_handlers")
class TestChatRoutesAuthenticateBeforeWork:
    """A valid body without good credentials must not reach Whisper, TTS or the LLM."""

    @pytest.mark.parametrize("path", TRANSCRIBE_ROUTES)
    def test_transcribe_routes_without_credentials_are_401(
        self, auth_env, agent_id, path
    ):
        response = auth_env.client.post(
            path, files=AUDIO, data={"data": json.dumps(_command(agent_id))}
        )
        assert response.status_code == 401

    @pytest.mark.parametrize("path", TRANSCRIBE_ROUTES)
    def test_transcribe_routes_with_wrong_body_key_are_403(
        self, auth_env, agent_id, valid_key, path
    ):
        response = auth_env.client.post(
            path,
            files=AUDIO,
            data={"data": json.dumps(_command(agent_id, "wrong-key"))},
        )
        assert response.status_code == 403

    @pytest.mark.parametrize("path", TRANSCRIBE_ROUTES)
    def test_transcribe_routes_with_wrong_bearer_key_are_403(
        self, auth_env, agent_id, valid_key, path
    ):
        response = auth_env.client.post(
            path,
            files=AUDIO,
            data={"data": json.dumps(_command(agent_id))},
            headers=_bearer("wrong-key"),
        )
        assert response.status_code == 403

    def test_ask_with_speech_without_credentials_is_401(self, auth_env, agent_id):
        response = auth_env.client.post(
            "/api/chat/askWithSpeech", json={"command": _command(agent_id)}
        )
        assert response.status_code == 401

    def test_ask_with_speech_with_wrong_key_is_403(self, auth_env, agent_id, valid_key):
        response = auth_env.client.post(
            "/api/chat/askWithSpeech",
            json={"command": _command(agent_id, "wrong-key")},
        )
        assert response.status_code == 403

    def test_valid_key_transcribes_then_answers(
        self, auth_env, agent_id, valid_key, monkeypatch
    ):
        monkeypatch.setattr(
            chat_routes,
            "transcribe_audio",
            lambda audio, language: {
                "success": True,
                "transcription": "hello there",
                "language": "en",
                "language_probability": 0.9,
            },
        )
        seen = {}

        def answer(command, agent):
            seen["question"] = command.chat_log[-1].content
            return {"response": f"answer from {agent.id}"}

        monkeypatch.setattr(chat_routes, "assemble_prompt_with_agent", answer)

        response = auth_env.client.post(
            "/api/chat/askTranscribe",
            files=AUDIO,
            data={"data": json.dumps(_command(agent_id, valid_key))},
        )

        assert response.status_code == 200
        assert response.json()["transcription"] == "hello there"
        assert response.json()["response"] == {"response": f"answer from {agent_id}"}
        assert seen["question"] == "hello there"
