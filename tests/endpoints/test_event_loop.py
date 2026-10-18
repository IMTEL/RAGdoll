"""Blocking chat work must not freeze the event loop for other requests.

The pipeline, Whisper and Piper are stubbed with a *blocking* ``time.sleep``.
If a handler ran that work on the event loop, a concurrent ``/ping`` could not
be served until the sleep ended.
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from src.main import app
from src.routes import chat as chat_routes
from tests.endpoints.conftest import AuthEnv, sample_agent


BLOCK_SECONDS = 1.5
PING_BUDGET_SECONDS = 0.5


@pytest.fixture
def started():
    """Set by a stub at the moment it starts blocking; holds the start time."""
    return SimpleNamespace(event=threading.Event(), at=None)


def _blocking(started, result):
    def _run(*_args, **_kwargs):
        started.at = time.monotonic()
        started.event.set()
        time.sleep(BLOCK_SECONDS)
        return result

    return _run


@pytest.fixture
def stubs(auth_env: AuthEnv, monkeypatch, started):
    agent_id = auth_env.agent_dao.add_agent(sample_agent()).id
    key = auth_env.access_service.generate_accesskey("k", None, agent_id).key
    monkeypatch.setattr(
        chat_routes,
        "assemble_prompt_with_agent",
        _blocking(started, {"response": "answer"}),
    )
    monkeypatch.setattr(
        chat_routes,
        "transcribe_audio",
        _blocking(
            started,
            {
                "success": True,
                "transcription": "hello",
                "language": "en",
                "language_probability": 1.0,
            },
        ),
    )
    speech = SimpleNamespace(to_dict=lambda: {"audio_base64": ""})
    tts = SimpleNamespace(
        synthesize=_blocking(started, speech),
        warmup=_blocking(started, {"success": True}),
    )
    monkeypatch.setattr(chat_routes, "get_tts_service", lambda: tts)
    monkeypatch.setattr(
        chat_routes, "warmup_whisper_model", _blocking(started, {"success": True})
    )
    return SimpleNamespace(agent_id=agent_id, key=key)


def _requests(stubs):
    command = {"agent_id": stubs.agent_id, "chat_log": [], "access_key": stubs.key}
    form = {"data": json.dumps(command)}
    audio = {"audio": ("a.wav", b"RIFF", "audio/wav")}
    return {
        "ask": {"url": "/api/chat/ask", "json": command},
        "askWithSpeech": {
            "url": "/api/chat/askWithSpeech",
            "json": {"command": command},
        },
        "askTranscribe": {
            "url": "/api/chat/askTranscribe",
            "data": form,
            "files": audio,
        },
        "askTranscribeWithSpeech": {
            "url": "/api/chat/askTranscribeWithSpeech",
            "data": form,
            "files": audio,
        },
        "transcribe": {
            "url": "/api/chat/transcribe",
            "files": audio,
            "headers": {
                "X-Agent-Id": stubs.agent_id,
                "Authorization": f"Bearer {stubs.key}",
            },
        },
        "tts": {
            "url": "/api/chat/tts",
            "json": {"text": "hi"},
            "headers": {
                "X-Agent-Id": stubs.agent_id,
                "Authorization": f"Bearer {stubs.key}",
            },
        },
        "tts/warmup": {
            "url": "/api/chat/tts/warmup",
            "headers": {
                "X-Agent-Id": stubs.agent_id,
                "Authorization": f"Bearer {stubs.key}",
            },
        },
        "stt/warmup": {
            "url": "/api/chat/stt/warmup",
            "headers": {
                "X-Agent-Id": stubs.agent_id,
                "Authorization": f"Bearer {stubs.key}",
            },
        },
    }


async def _slow_request_and_ping(started, request: dict):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        async def slow():
            begin = time.monotonic()
            response = await client.post(**request)
            return response, time.monotonic() - begin

        async def ping():
            # Wait until the slow handler is inside its blocking work.
            while not started.event.is_set():
                await asyncio.sleep(0.01)
            response = await client.get("/ping")
            return response, time.monotonic() - started.at

        return await asyncio.gather(slow(), ping())


@pytest.mark.parametrize(
    "route",
    [
        "ask",
        "askWithSpeech",
        "askTranscribe",
        "askTranscribeWithSpeech",
        "transcribe",
        "tts",
        "tts/warmup",
        "stt/warmup",
    ],
)
def test_ping_is_served_while_chat_work_blocks(stubs, started, route, monkeypatch):
    monkeypatch.setattr(chat_routes.config, "PUBLIC_SPEECH_ENDPOINTS", False)
    request = _requests(stubs)[route]

    (slow, slow_elapsed), (ping, ping_after_start) = asyncio.run(
        _slow_request_and_ping(started, request)
    )

    assert slow.status_code == 200, slow.text
    assert ping.status_code == 200
    assert slow_elapsed >= BLOCK_SECONDS
    assert ping_after_start < PING_BUDGET_SECONDS
