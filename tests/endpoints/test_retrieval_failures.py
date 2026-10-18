"""Retrieval failures are surfaced instead of silently becoming empty context."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pymongo.errors import PyMongoError

from src import pipeline
from src.models.agent import Agent, Role
from src.models.errors import EmbeddingAPIError, EmbeddingError
from src.rag_service.dao.context import hybrid_search as hybrid_search_module
from src.rag_service.dao.context.faiss_vector_store import (
    FAISSVectorStoreManager,
    IndexDimensionMismatchError,
)
from src.rag_service.dao.context.mongodb_context_dao import MongoDBContextDAO
from src.routes import chat as chat_routes
from tests.endpoints.conftest import AuthEnv, sample_agent


def _rules(messages) -> str:
    """The first system message, where behaviour rules live."""
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


def _all_text(messages) -> str:
    return "\n".join(m["content"] for m in messages)


ROLE = Role(name="student", description="d", document_access=["doc1"])


class _FailingEmbedding:
    def __init__(self, error=None):
        self.error = error or EmbeddingAPIError("Gemini", "models/x", "quota exceeded")
        self.calls = 0

    def get_embedding(self, text, **kwargs):
        self.calls += 1
        raise self.error


@pytest.fixture
def fake_llm(monkeypatch):
    llm = MagicMock()
    llm.generate.return_value = "I am not sure."
    monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: llm)
    return llm


@pytest.fixture
def real_pipeline(monkeypatch):
    """Undo the endpoint conftest stub so the real pipeline runs."""
    monkeypatch.setattr(
        chat_routes, "assemble_prompt_with_agent", pipeline.assemble_prompt_with_agent
    )


def _add_agent(auth_env: AuthEnv, **fields) -> tuple[str, str]:
    agent = sample_agent()
    agent.context_aware_retrieval = False
    agent.roles = [ROLE]
    for name, value in fields.items():
        setattr(agent, name, value)
    agent_id = auth_env.agent_dao.add_agent(agent).id
    key = auth_env.access_service.generate_accesskey("k", None, agent_id).key
    return agent_id, key


def _ask(
    auth_env: AuthEnv,
    agent_id: str,
    key: str,
    role: str | None = "student",
    content: str = "What is the boiling point?",
):
    return auth_env.client.post(
        "/api/chat/ask",
        json={
            "agent_id": agent_id,
            "access_key": key,
            "active_role_id": role,
            "chat_log": [{"role": "user", "content": content}],
        },
    )


class TestRetrievalErrorPolicy:
    def test_agent_defaults_to_fail(self):
        assert sample_agent().on_retrieval_error == "fail"

    def test_stored_agent_without_field_defaults_to_fail(self):
        data = sample_agent().model_dump()
        data.pop("on_retrieval_error", None)
        assert Agent.model_validate(data).on_retrieval_error == "fail"

    def test_embedding_failure_under_fail_is_503(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        failing = _FailingEmbedding(EmbeddingError("embedding service down"))
        monkeypatch.setattr(
            pipeline, "create_embeddings_model", lambda *a, **k: failing
        )
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: MagicMock())
        agent_id, key = _add_agent(auth_env)

        r = _ask(auth_env, agent_id, key)

        assert r.status_code == 503
        assert r.json()["detail"] == "Knowledge retrieval failed"
        fake_llm.generate.assert_not_called()

    def test_embedding_failure_answer_without_context(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        monkeypatch.setattr(
            pipeline, "create_embeddings_model", lambda *a, **k: _FailingEmbedding()
        )
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: MagicMock())
        agent_id, key = _add_agent(
            auth_env, on_retrieval_error="answer_without_context"
        )

        r = _ask(auth_env, agent_id, key)

        assert r.status_code == 200, r.text
        body = r.json()["response"]
        assert body["retrieval"]["status"] == "error"
        assert body["retrieval"]["error"]
        assert "quota" not in body["retrieval"]["error"]
        assert body["metadata"]["num_context_retrieved"] == 0
        prompt = _rules(fake_llm.generate.call_args.args[0])
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION_DOCUMENTS_FIRST in prompt


class TestRetrievalSkipsAndKeyErrors:
    """Retrieval is skipped when there is nothing to search.

    Blank queries and roles without documents never call the embedder, and
    provider key problems surface as the provider's status code.
    """

    def _embedder(self, monkeypatch, failing=None):
        embedder = failing or MagicMock()
        if failing is None:
            embedder.get_embedding.return_value = [0.1, 0.2]
        monkeypatch.setattr(
            pipeline, "create_embeddings_model", lambda *a, **k: embedder
        )
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: MagicMock())
        return embedder

    def test_blank_query_is_empty_without_embedding(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        embedder = self._embedder(monkeypatch, _FailingEmbedding())
        agent_id, key = _add_agent(auth_env)

        r = _ask(auth_env, agent_id, key, content="   ")

        assert r.status_code == 200, r.text
        assert r.json()["response"]["retrieval"] == {"status": "empty", "error": None}
        assert embedder.calls == 0
        fake_llm.generate.assert_called_once()

    def test_role_without_documents_is_empty_without_embedding(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        embedder = self._embedder(monkeypatch, _FailingEmbedding())
        agent_id, key = _add_agent(auth_env)

        r = _ask(auth_env, agent_id, key, role=None)

        assert r.status_code == 200, r.text
        assert r.json()["response"]["retrieval"] == {"status": "empty", "error": None}
        assert embedder.calls == 0

    @pytest.mark.parametrize(
        ("message", "status"),
        [
            ("API key not valid", 401),
            ("quota exceeded for this project", 429),
            ("rate limit reached", 429),
        ],
    )
    def test_embedding_key_or_quota_error_uses_provider_status(
        self, auth_env, real_pipeline, fake_llm, monkeypatch, message, status
    ):
        error = EmbeddingAPIError("Gemini", "models/x", Exception(message))
        self._embedder(monkeypatch, _FailingEmbedding(error))
        agent_id, key = _add_agent(auth_env)

        r = _ask(auth_env, agent_id, key)

        assert r.status_code == status, r.text
        assert r.json()["detail"] != "Knowledge retrieval failed"
        fake_llm.generate.assert_not_called()

    def test_embedding_error_status_from_provider_exception(self):
        class _Provider402Error(Exception):
            status_code = 402

        error = EmbeddingAPIError(
            "OpenAI", "m", _Provider402Error("insufficient credits")
        )
        assert error.status_code == 402

    def test_invalid_stored_embedding_key_is_401(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        embedder = self._embedder(monkeypatch)
        agent_id, key = _add_agent(auth_env)
        monkeypatch.setattr(
            pipeline,
            "resolve_provider_keys",
            lambda agent, _dao: agent.model_copy(
                update={"embedding_api_key": "", "embedding_api_key_invalid": True}
            ),
        )

        r = _ask(auth_env, agent_id, key)

        assert r.status_code == 401, r.text
        embedder.get_embedding.assert_not_called()
        fake_llm.generate.assert_not_called()

    def test_invalid_stored_llm_key_is_401(
        self, auth_env, real_pipeline, fake_llm, monkeypatch
    ):
        embedder = self._embedder(monkeypatch)
        agent_id, key = _add_agent(auth_env)
        monkeypatch.setattr(
            pipeline,
            "resolve_provider_keys",
            lambda agent, _dao: agent.model_copy(
                update={"llm_api_key": "", "llm_api_key_invalid": True}
            ),
        )

        r = _ask(auth_env, agent_id, key)

        assert r.status_code == 401, r.text
        embedder.get_embedding.assert_not_called()
        fake_llm.generate.assert_not_called()


class TestRetrievalStatus:
    def _run(self, monkeypatch, dao, agent=None):
        agent = agent or sample_agent()
        agent.id = "a1"
        agent.context_aware_retrieval = False
        agent.roles = [ROLE]
        llm = MagicMock()
        llm.generate.return_value = "answer"
        monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: llm)
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: dao)
        embeddings = MagicMock()
        embeddings.get_embedding.return_value = [0.1, 0.2]
        monkeypatch.setattr(
            pipeline, "create_embeddings_model", lambda *a, **k: embeddings
        )
        from src.models.chat.command import Command

        command = Command(
            agent_id="a1",
            active_role_id="student",
            chat_log=[{"role": "user", "content": "hello"}],
        )
        return pipeline.assemble_prompt_with_agent(command, agent), llm

    def test_no_matches_is_empty_and_generation_continues(self, monkeypatch):
        dao = MagicMock()
        dao.get_context_for_agent.return_value = []
        result, llm = self._run(monkeypatch, dao)
        assert result["retrieval"] == {"status": "empty", "error": None}
        assert result["metadata"]["num_context_retrieved"] == 0
        prompt = _all_text(llm.generate.call_args.args[0])
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION not in prompt
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION_DOCUMENTS_FIRST not in prompt

    def test_matches_are_ok(self, monkeypatch):
        dao = MagicMock()
        dao.get_context_for_agent.return_value = [
            SimpleNamespace(text="water boils", document_name="d", chunk_index=0)
        ]
        result, _ = self._run(monkeypatch, dao)
        assert result["retrieval"] == {"status": "ok", "error": None}
        assert result["metadata"]["num_context_retrieved"] == 1

    def test_index_dimension_mismatch_is_retrieval_error(self, monkeypatch):
        dao = MagicMock()
        dao.get_context_for_agent.side_effect = IndexDimensionMismatchError(
            "a1", 384, 2
        )
        with pytest.raises(pipeline.RetrievalFailedError):
            self._run(monkeypatch, dao)

    def test_dao_error_answer_without_context(self, monkeypatch):
        dao = MagicMock()
        dao.get_context_for_agent.side_effect = PyMongoError("conn refused")
        agent = sample_agent()
        agent.on_retrieval_error = "answer_without_context"
        result, llm = self._run(monkeypatch, dao, agent)
        assert result["retrieval"]["status"] == "error"
        assert "PyMongoError" in result["retrieval"]["error"]
        prompt = _rules(llm.generate.call_args.args[0])
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION_DOCUMENTS_FIRST in prompt

    def test_dao_error_documents_only_uses_strict_instruction(self, monkeypatch):
        dao = MagicMock()
        dao.get_context_for_agent.side_effect = PyMongoError("conn refused")
        agent = sample_agent()
        agent.on_retrieval_error = "answer_without_context"
        agent.knowledge_scope = "documents_only"
        _, llm = self._run(monkeypatch, dao, agent)
        prompt = _rules(llm.generate.call_args.args[0])
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION in prompt
        assert pipeline.NO_REFERENCE_MATERIAL_INSTRUCTION_DOCUMENTS_FIRST not in prompt


def _raise_mongo(*args, **kwargs):
    raise PyMongoError("connection refused")


@pytest.fixture
def unreachable_mongo_dao(monkeypatch):
    """Real MongoDBContextDAO + hybrid_search whose collection is unreachable."""
    dao = MongoDBContextDAO()
    for method in ("find", "find_one", "aggregate"):
        monkeypatch.setattr(dao.collection, method, _raise_mongo)
    monkeypatch.setattr(pipeline, "get_context_dao", lambda: dao)
    embeddings = MagicMock()
    embeddings.get_embedding.return_value = [0.1, 0.2]
    monkeypatch.setattr(pipeline, "create_embeddings_model", lambda *a, **k: embeddings)
    return dao


class TestMongoUnreachableDuringSearch:
    def test_keyword_search_mongo_error_is_503_under_fail(
        self, auth_env, real_pipeline, fake_llm, unreachable_mongo_dao
    ):
        agent_id, key = _add_agent(auth_env)
        r = _ask(auth_env, agent_id, key, role="student")
        assert r.status_code == 503
        assert r.json()["detail"] == "Knowledge retrieval failed"
        fake_llm.generate.assert_not_called()

    def test_mongo_error_is_status_error_not_empty(
        self, auth_env, real_pipeline, fake_llm, unreachable_mongo_dao
    ):
        agent_id, key = _add_agent(
            auth_env,
            on_retrieval_error="answer_without_context",
        )
        r = _ask(auth_env, agent_id, key, role="student")
        assert r.status_code == 200, r.text
        assert r.json()["response"]["retrieval"]["status"] == "error"
        assert "PyMongoError" in r.json()["response"]["retrieval"]["error"]

    def test_vector_metadata_lookup_mongo_error_propagates(
        self, unreachable_mongo_dao, monkeypatch
    ):
        monkeypatch.setattr(
            FAISSVectorStoreManager, "search", lambda self, *a, **k: {"c1": 0.9}
        )
        with pytest.raises(PyMongoError):
            hybrid_search_module.hybrid_search(
                1.0,
                "a1",
                [0.1, 0.2],
                "q",
                "q",
                ["doc1"],
                unreachable_mongo_dao.collection,
                0.0,
            )

    def test_missing_vector_row_is_still_skipped(self, monkeypatch):
        dao = MongoDBContextDAO()
        monkeypatch.setattr(
            FAISSVectorStoreManager, "search", lambda self, *a, **k: {"c1": 0.9}
        )
        assert (
            hybrid_search_module.hybrid_search(
                1.0, "a1", [0.1, 0.2], "q", "q", ["doc1"], dao.collection, 0.0
            )
            == []
        )
