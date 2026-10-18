"""Cheaper retrieval-query rewrite and client reuse."""

import logging
from unittest.mock import MagicMock

import pytest

from src import llm as llm_module
from src import pipeline
from src.models.chat.command import Command
from src.models.chat.message import Message
from src.rag_service import embeddings as emb_module
from tests.endpoints.conftest import sample_agent


@pytest.fixture(autouse=True)
def _clear_caches():
    llm_module.clear_llm_cache()
    emb_module.clear_embedding_cache()
    yield
    llm_module.clear_llm_cache()
    emb_module.clear_embedding_cache()


def _run(monkeypatch, messages):
    agent = sample_agent()
    agent.id = "a1"
    agent.context_aware_retrieval = True
    agent.prompt = "PERSONA-SENTINEL"
    fake_llm = MagicMock()
    fake_llm.generate.return_value = "standalone"
    monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: fake_llm)
    dao = MagicMock()
    dao.get_context_for_agent.return_value = []
    monkeypatch.setattr(pipeline, "get_context_dao", lambda: dao)
    emb = MagicMock()
    emb.get_embedding.return_value = [0.0]
    monkeypatch.setattr(pipeline, "create_embeddings_model", lambda *a, **k: emb)
    command = Command(agent_id="a1", chat_log=messages)
    pipeline.assemble_prompt_with_agent(command, agent)
    return fake_llm


def test_first_turn_makes_no_rewrite_call(monkeypatch):
    llm = _run(monkeypatch, [Message(role="user", content="What is it?")])
    assert llm.generate.call_count == 1  # main generate only


def test_no_back_reference_makes_no_rewrite_call(monkeypatch):
    llm = _run(
        monkeypatch,
        [
            Message(role="user", content="Hello"),
            Message(role="agent", content="Hi"),
            Message(role="user", content="Tell me about dragons"),
        ],
    )
    assert llm.generate.call_count == 1


def test_back_reference_rewrites_once_with_bounded_history(monkeypatch):
    msgs = []
    for i in range(5):
        msgs.append(Message(role="user", content=f"OLDTURN{i}"))
        msgs.append(Message(role="agent", content=f"OLDREPLY{i}"))
    msgs.append(Message(role="user", content="How big is it?"))
    llm = _run(monkeypatch, msgs)
    assert llm.generate.call_count == 2
    first = llm.generate.call_args_list[0]
    messages = first.args[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    prompt = "\n".join(m["content"] for m in messages)
    assert first.args[1] == 64
    assert "PERSONA-SENTINEL" not in prompt
    assert "OLDTURN3" not in prompt and "OLDREPLY2" not in prompt
    assert "OLDREPLY4" in prompt and "How big is it?" in prompt


def test_back_reference_is_whole_word_case_insensitive():
    assert pipeline._needs_query_rewrite(
        Command(
            agent_id="a",
            chat_log=[
                Message(role="user", content="x"),
                Message(role="user", content="Hva er DET?"),
            ],
        )
    )
    assert not pipeline._needs_query_rewrite(
        Command(
            agent_id="a",
            chat_log=[
                Message(role="user", content="x"),
                Message(role="user", content="Tell me about Italy within reason"),
            ],
        )
    )


@pytest.mark.parametrize(
    "word",
    ["them", "these", "dem", "disse", "denne", "his", "her", "hers", "its", "their"],
)
def test_extended_back_reference_words_trigger_rewrite(word):
    command = Command(
        agent_id="a",
        chat_log=[
            Message(role="user", content="earlier"),
            Message(role="user", content=f"Tell me about {word} please"),
        ],
    )
    assert pipeline._needs_query_rewrite(command)


def test_timing_log_has_durations_only(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="src.pipeline"):
        _run(monkeypatch, [Message(role="user", content="SECRETWORD here")])
    lines = [r.getMessage() for r in caplog.records if "timings" in r.getMessage()]
    assert len(lines) == 1
    for stage in ("rewrite", "embed", "retrieve", "generate"):
        assert stage in lines[0]
    assert "SECRETWORD" not in caplog.text


def test_llm_client_constructed_once_per_key(monkeypatch):
    built = []

    class Fake:
        def __init__(self, model=None, api_key=None):
            built.append((model, api_key))

    monkeypatch.setattr(llm_module, "OpenAILLM", Fake)
    a = llm_module.create_llm("openai", "m", "key1")
    b = llm_module.create_llm("openai", "m", "key1")
    assert a is b and len(built) == 1
    c = llm_module.create_llm("openai", "m", "key2")  # rotated key
    assert c is not a and len(built) == 2
    llm_module.create_llm("openai", "other", "key1")
    assert len(built) == 3
    for key in llm_module._llm_cache._items:
        assert "key1" not in key and "key2" not in key


def test_embedding_client_constructed_once(monkeypatch):
    built = []

    class Fake:
        def __init__(self, model_name=None, embedding_api_key=None):
            built.append(model_name)

    monkeypatch.setattr(emb_module, "OpenAIEmbedding", Fake)
    a = emb_module.create_embeddings_model("openai:m", embedding_api_key="k")
    b = emb_module.create_embeddings_model("openai:m", embedding_api_key="k")
    assert a is b and len(built) == 1
    emb_module.create_embeddings_model("openai:m", embedding_api_key="k2")
    assert len(built) == 2


def test_cache_is_bounded():
    from src.utils.client_cache import ClientCache

    cache = ClientCache(maxsize=2)
    for i in range(5):
        cache.get_or_create(i, lambda: object())
    assert len(cache) == 2
