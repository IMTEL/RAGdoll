"""Gemini classes use the installed google-genai client API (offline, mocked)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import llm as llm_module
from src.models.errors import EmbeddingAPIError, LLMAPIError, LLMGenerationError
from src.rag_service import embeddings as embeddings_module


@pytest.fixture
def fake_client(monkeypatch):
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr(embeddings_module.genai, "Client", factory)
    return SimpleNamespace(client=client, factory=factory)


class TestGoogleEmbedding:
    def test_constructs_with_dummy_key(self, fake_client):
        model = embeddings_module.GoogleEmbedding(embedding_api_key="dummy")
        fake_client.factory.assert_called_once()
        assert fake_client.factory.call_args.kwargs["api_key"] == "dummy"
        assert model.model_name == "models/text-embedding-004"

    def test_returns_list_of_floats(self, fake_client):
        fake_client.client.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )
        model = embeddings_module.GoogleEmbedding(embedding_api_key="dummy")
        assert model.get_embedding("hello\nworld") == [0.1, 0.2, 0.3]
        kwargs = fake_client.client.models.embed_content.call_args.kwargs
        assert kwargs["model"] == "models/text-embedding-004"
        assert kwargs["contents"] == "hello world"

    def test_quota_error_maps_to_embedding_api_error(self, fake_client):
        fake_client.client.models.embed_content.side_effect = RuntimeError(
            "429 quota exceeded"
        )
        model = embeddings_module.GoogleEmbedding(embedding_api_key="dummy")
        with pytest.raises(EmbeddingAPIError):
            model.get_embedding("hello")

    def test_list_models_uses_client(self, fake_client):
        fake_client.client.models.list.return_value = [
            SimpleNamespace(
                name="models/text-embedding-004", supported_actions=["embedContent"]
            ),
            SimpleNamespace(
                name="models/gemini-2.0-flash", supported_actions=["generateContent"]
            ),
        ]
        assert embeddings_module.list_gemini_embedding_models("dummy") == [
            "gemini:models/text-embedding-004"
        ]


class TestGeminiLLM:
    def test_constructs_with_dummy_key(self, fake_client):
        model = llm_module.GeminiLLM(model="gemini-2.0-flash", api_key="dummy")
        fake_client.factory.assert_called_once()
        assert fake_client.factory.call_args.kwargs["api_key"] == "dummy"
        assert model.model == "gemini-2.0-flash"

    def test_generate_returns_text(self, fake_client):
        fake_client.client.models.generate_content.return_value = SimpleNamespace(
            text="  hello  "
        )
        model = llm_module.GeminiLLM(model="gemini-2.0-flash", api_key="dummy")
        assert model.generate("hi", 50, 0.4) == "hello"
        kwargs = fake_client.client.models.generate_content.call_args.kwargs
        assert kwargs["model"] == "gemini-2.0-flash"
        contents = kwargs["contents"]
        assert [c.role for c in contents] == ["user"]
        assert contents[0].parts[0].text == "hi"
        config = kwargs["config"]
        assert config.max_output_tokens == 50
        assert config.temperature == 0.4

    def test_none_text_is_generation_error(self, fake_client):
        fake_client.client.models.generate_content.return_value = SimpleNamespace(
            text=None
        )
        model = llm_module.GeminiLLM(model="gemini-2.0-flash", api_key="dummy")
        with pytest.raises(LLMGenerationError):
            model.generate("hi")

    def test_auth_error_maps_to_llm_api_error(self, fake_client):
        fake_client.client.models.generate_content.side_effect = RuntimeError(
            "API key not valid"
        )
        model = llm_module.GeminiLLM(model="gemini-2.0-flash", api_key="dummy")
        with pytest.raises(LLMAPIError) as exc:
            model.generate("hi")
        assert exc.value.status_code == 401

    def test_list_models_uses_client(self, fake_client):
        fake_client.client.models.list.return_value = [
            SimpleNamespace(name="models/gemini-2.0-flash", description="d"),
        ]
        names = [m.name for m in llm_module.list_gemini_models("dummy")]
        assert "models/gemini-2.0-flash" in names


def _embed_response():
    return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])


def _task_type(fake_client) -> str:
    return fake_client.client.models.embed_content.call_args.kwargs["config"].task_type


class TestEmbeddingTaskType:
    def test_default_is_retrieval_query(self, fake_client):
        fake_client.client.models.embed_content.return_value = _embed_response()
        embeddings_module.GoogleEmbedding(embedding_api_key="k").get_embedding("q")
        assert _task_type(fake_client) == "RETRIEVAL_QUERY"

    def test_document_task_is_retrieval_document(self, fake_client):
        fake_client.client.models.embed_content.return_value = _embed_response()
        embeddings_module.GoogleEmbedding(embedding_api_key="k").get_embedding(
            "d", task="document"
        )
        assert _task_type(fake_client) == "RETRIEVAL_DOCUMENT"

    def test_ingestion_embeds_as_document(self, fake_client):
        from src import context_upload

        fake_client.client.models.embed_content.return_value = _embed_response()
        context_upload.compute_embedding(
            "chunk text", "gemini:models/text-embedding-004", "k"
        )
        assert _task_type(fake_client) == "RETRIEVAL_DOCUMENT"

    def test_pipeline_query_embeds_as_query(self, fake_client, monkeypatch):
        from src import pipeline
        from src.models.chat.command import Command
        from tests.endpoints.conftest import sample_agent

        fake_client.client.models.embed_content.return_value = _embed_response()
        dao = MagicMock()
        dao.get_context_for_agent.return_value = []
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: dao)
        llm = MagicMock()
        llm.generate.return_value = "answer"
        monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: llm)
        agent = sample_agent()
        agent.id = "a1"
        agent.embedding_model = "gemini:models/text-embedding-004"
        agent.context_aware_retrieval = False

        pipeline.assemble_prompt_with_agent(
            Command(agent_id="a1", chat_log=[{"role": "user", "content": "hi"}]),
            agent,
        )
        assert _task_type(fake_client) == "RETRIEVAL_QUERY"

    def test_other_providers_accept_and_ignore_task(self, monkeypatch):
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.5])]
        )
        monkeypatch.setattr(embeddings_module.openai, "Client", lambda **k: client)
        for cls in (embeddings_module.OpenAIEmbedding, embeddings_module.IdunEmbedding):
            model = cls(embedding_api_key="k")
            assert model.get_embedding("d", task="document") == [0.5]
            assert "task" not in client.embeddings.create.call_args.kwargs
