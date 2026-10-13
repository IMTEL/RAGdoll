"""The chat pipeline answers with the temperature configured on the agent."""

import src.pipeline as pipeline
from src.models import Agent
from src.models.chat.command import Command
from src.models.chat.message import Message


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_tokens=None, temperature=None):
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature})
        return "Hei!"


class _NoEmbeddings:
    def get_embedding(self, text):
        return [0.0]


class _NoContexts:
    def get_context_for_agent(self, **kwargs):
        return []


def _agent(**overrides) -> Agent:
    fields = {
        "id": "agent-1",
        "name": "Intervjuer",
        "description": "",
        "prompt": "Du er en intervjuer.",
        "llm_model": "openai/gpt-oss-120b",
        "llm_api_key": "key",
        "embedding_model": "idun:Qwen/Qwen3-Embedding-8B",
        "embedding_api_key": "key",
        "last_updated": "2026-10-13T00:00:00Z",
        "context_aware_retrieval": False,
    }
    fields.update(overrides)
    return Agent(**fields)


def _ask(monkeypatch, agent: Agent) -> _RecordingLLM:
    llm = _RecordingLLM()
    monkeypatch.setattr(pipeline, "create_llm", lambda **kwargs: llm)
    monkeypatch.setattr(
        pipeline, "create_embeddings_model", lambda *a, **k: _NoEmbeddings()
    )
    monkeypatch.setattr(pipeline, "get_context_dao", lambda: _NoContexts())

    command = Command(
        agent_id="agent-1",
        chat_log=[Message(role="user", content="Hei")],
        include_progress=False,
    )
    pipeline.assemble_prompt_with_agent(command, agent)
    return llm


def test_answer_uses_the_agents_temperature(monkeypatch):
    llm = _ask(monkeypatch, _agent(llm_temperature=0.2))

    assert llm.calls[-1]["temperature"] == 0.2


def test_answer_uses_the_default_temperature_when_none_is_set(monkeypatch):
    agent = _agent()
    llm = _ask(monkeypatch, agent)

    assert llm.calls[-1]["temperature"] == agent.llm_temperature
