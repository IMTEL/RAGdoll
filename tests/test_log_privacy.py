"""Prompts, conversations and secrets must not reach logs or stdout."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src import pipeline
from src.log_redaction import redact
from src.models.chat.command import Command
from src.models.chat.message import Message
from tests.endpoints.conftest import sample_agent


CHAT_SENTINEL = "SENTINEL-CHAT-7f3a91"
CHUNK_SENTINEL = "SENTINEL-CHUNK-b24c10"


def _run_pipeline(monkeypatch):
    agent = sample_agent()
    agent.id = "a1"
    agent.context_aware_retrieval = True

    fake_llm = MagicMock()
    fake_llm.generate.return_value = "plain answer"
    monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: fake_llm)

    dao = MagicMock()
    dao.get_context_for_agent.return_value = [
        SimpleNamespace(
            text=f"doc text {CHUNK_SENTINEL}", document_name="d", chunk_index=0
        )
    ]
    monkeypatch.setattr(pipeline, "get_context_dao", lambda: dao)
    embeddings = MagicMock()
    embeddings.get_embedding.return_value = [0.0]
    monkeypatch.setattr(pipeline, "create_embeddings_model", lambda *a, **k: embeddings)
    command = Command(
        agent_id="a1",
        chat_log=[
            Message(role="user", content=f"hello {CHAT_SENTINEL}"),
            Message(role="agent", content="hi"),
            Message(role="user", content=f"more {CHAT_SENTINEL}"),
        ],
    )
    return pipeline.assemble_prompt_with_agent(command, agent)


def test_pipeline_does_not_log_prompt_or_chunks(monkeypatch, caplog, capsys):
    monkeypatch.setattr(pipeline.config, "LOG_CONVERSATIONS", False, raising=False)
    with caplog.at_level(logging.DEBUG):
        _run_pipeline(monkeypatch)
    out = capsys.readouterr()
    for sentinel in (CHAT_SENTINEL, CHUNK_SENTINEL):
        assert sentinel not in caplog.text
        assert sentinel not in out.out
        assert sentinel not in out.err


class TestRedact:
    @pytest.mark.parametrize(
        "secret",
        [
            "rgd_abcdef1234567890",
            "sk-abcdefghijklmnop1234",
            "Bearer eyJhbGciOi.abc-def_ghi",
            "a" * 40,
            "0123456789abcdef" * 3,
            "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=",
        ],
    )
    def test_masks_secrets(self, secret):
        out = redact(f"before {secret} after")
        assert secret not in out
        assert "before" in out and "after" in out

    def test_leaves_normal_text(self):
        text = "Hello, what is the capital of Norway?"
        assert redact(text) == text


def test_logging_config_defaults(monkeypatch):
    from src.constants import build_logging_config

    monkeypatch.delenv("LOG_FILE", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    cfg = build_logging_config()
    assert cfg["root"]["level"] == "INFO"
    assert cfg["root"]["handlers"] == ["console"]
    assert cfg["loggers"]["pymongo"]["level"] == "WARNING"

    monkeypatch.setenv("LOG_FILE", "x.log")
    cfg = build_logging_config()
    assert cfg["handlers"]["file"]["class"].endswith("RotatingFileHandler")


def test_command_validation_error_does_not_log_input(caplog, capsys):
    import json

    from src.models.chat.command import (
        command_from_json,
        command_from_json_transcribe_version,
    )

    key = "rgd_SECRETKEY1234567890"
    payload = json.dumps(
        {
            "agent_id": "a1",
            "access_key": key,
            "chat_log": [
                {"role": "user", "content": CHAT_SENTINEL, "bad": 1},
                {"role": CHAT_SENTINEL},
            ],
        }
    )
    with caplog.at_level(logging.DEBUG):
        assert command_from_json(payload) is None
        assert command_from_json_transcribe_version(payload, "q") is None
    out = capsys.readouterr()
    for text in (caplog.text, out.out, out.err):
        assert CHAT_SENTINEL not in text
        assert "SECRETKEY" not in text
    assert "Command validation failed" in caplog.text


def test_log_conversations_logs_and_masks(monkeypatch, caplog):
    monkeypatch.setattr(pipeline.config, "LOG_CONVERSATIONS", True, raising=False)
    token = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0U1v="
    monkeypatch.setattr(
        pipeline, "create_llm", lambda *a, **k: MagicMock(generate=lambda *x: "ok")
    )
    agent = sample_agent()
    agent.id = "a1"
    agent.context_aware_retrieval = False
    monkeypatch.setattr(pipeline, "get_context_dao", lambda: MagicMock())
    emb = MagicMock()
    emb.get_embedding.return_value = [0.0]
    monkeypatch.setattr(pipeline, "create_embeddings_model", lambda *a, **k: emb)
    command = Command(
        agent_id="a1",
        chat_log=[
            Message(
                role="user", content=f"hello {CHAT_SENTINEL} rgd_abc123def456 {token}"
            )
        ],
    )
    with caplog.at_level(logging.INFO):
        pipeline.assemble_prompt_with_agent(command, agent)
    assert CHAT_SENTINEL in caplog.text
    assert "rgd_abc123def456" not in caplog.text
    assert token not in caplog.text
