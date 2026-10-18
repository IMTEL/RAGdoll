"""Correctness tests for chat error handling, speech responses, LLM and pipeline."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import llm as llm_module
from src import pipeline
from src.models.chat.command import Command
from src.models.errors import LLMAPIError, LLMGenerationError
from src.routes import chat as chat_routes
from tests.endpoints.conftest import AuthEnv, sample_agent


@pytest.fixture
def agent_id(auth_env: AuthEnv) -> str:
    return auth_env.agent_dao.add_agent(sample_agent()).id


@pytest.fixture
def key(auth_env: AuthEnv, agent_id: str) -> str:
    return auth_env.access_service.generate_accesskey("k", None, agent_id).key


def _raiser(exc):
    def _raise(command, agent):
        raise exc

    return _raise


class TestAskErrors:
    def test_generation_error_keeps_status_and_string_detail(
        self, auth_env, agent_id, key, monkeypatch
    ):
        err = LLMGenerationError("openai", "gpt-4", "boom")
        monkeypatch.setattr(chat_routes, "assemble_prompt_with_agent", _raiser(err))
        r = auth_env.client.post(
            "/api/chat/ask",
            json={"agent_id": agent_id, "chat_log": [], "access_key": key},
        )
        assert r.status_code == err.status_code
        assert isinstance(r.json()["detail"], str)
        assert "boom" in r.json()["detail"]

    def test_api_error_keeps_status_and_string_detail(
        self, auth_env, agent_id, key, monkeypatch
    ):
        err = LLMAPIError("openai", "gpt-4", "quota", "slow down", 429)
        monkeypatch.setattr(chat_routes, "assemble_prompt_with_agent", _raiser(err))
        r = auth_env.client.post(
            "/api/chat/ask",
            json={"agent_id": agent_id, "chat_log": [], "access_key": key},
        )
        assert r.status_code == 429
        assert isinstance(r.json()["detail"], str)

    def test_unexpected_error_is_500_with_string_detail(
        self, auth_env, agent_id, key, monkeypatch
    ):
        monkeypatch.setattr(
            chat_routes, "assemble_prompt_with_agent", _raiser(RuntimeError("x"))
        )
        r = auth_env.client.post(
            "/api/chat/ask",
            json={"agent_id": agent_id, "chat_log": [], "access_key": key},
        )
        assert r.status_code == 500
        assert isinstance(r.json()["detail"], str)


class TestAskTranscribeWithSpeech:
    def test_returns_single_object_with_speech_of_answer_text(
        self, auth_env, agent_id, key, monkeypatch
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
        spoken = []

        class FakeSpeech:
            def to_dict(self):
                return {"audio": "b64"}

        class FakeTTS:
            def synthesize(self, text, language=None):
                spoken.append(text)
                return FakeSpeech()

        monkeypatch.setattr(chat_routes, "get_tts_service", lambda: FakeTTS())

        r = auth_env.client.post(
            "/api/chat/askTranscribeWithSpeech",
            files={"audio": ("a.wav", io.BytesIO(b"x"), "audio/wav")},
            data={"data": f'{{"agent_id": "{agent_id}", "access_key": "{key}"}}'},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body, dict)
        assert body["transcription"] == "hello there"
        assert body["response"] == {"response": f"answer from {agent_id}"}
        assert body["speech"] == {"audio": "b64"}
        assert spoken == [f"answer from {agent_id}"]


class TestOpenAILLM:
    def _llm(self, content):
        instance = llm_module.OpenAILLM.__new__(llm_module.OpenAILLM)
        instance.model = "gpt-4"
        instance.client = MagicMock()
        instance.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )
        return instance

    def test_messages_use_content_key(self):
        instance = self._llm(" hi ")
        assert instance.generate("question", 10, 0.2) == "hi"
        kwargs = instance.client.chat.completions.create.call_args.kwargs
        for message in kwargs["messages"]:
            assert "content" in message
            assert "text" not in message
        assert kwargs["messages"][-1] == {"role": "user", "content": "question"}

    def test_none_content_raises_generation_error(self):
        with pytest.raises(LLMGenerationError):
            self._llm(None).generate("question")


class TestPipelinePassesLLMParams:
    def test_generate_gets_max_tokens_and_temperature(self, monkeypatch):
        agent = sample_agent()
        agent.id = "a1"
        agent.llm_max_tokens = 123
        agent.llm_temperature = 0.3
        agent.context_aware_retrieval = False

        fake_llm = MagicMock()
        fake_llm.generate.return_value = "plain answer"
        monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: fake_llm)
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: MagicMock())
        embeddings = MagicMock()
        embeddings.get_embedding.return_value = [0.0]
        monkeypatch.setattr(
            pipeline, "create_embeddings_model", lambda *a, **k: embeddings
        )

        command = Command(agent_id="a1", chat_log=[])
        pipeline.assemble_prompt_with_agent(command, agent)

        args, kwargs = fake_llm.generate.call_args
        passed = list(args[1:]) + list(kwargs.values())
        assert 123 in passed
        assert 0.3 in passed
