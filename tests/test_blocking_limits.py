"""Outbound LLM/embedding calls time out; Whisper and Piper work is bounded."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src import llm as llm_module
from src import transcribe as transcribe_module
from src import tts as tts_module
from src.config import Config
from src.rag_service import embeddings as embeddings_module


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.LLM_TIMEOUT_SECONDS == 60
        assert config.STT_MAX_CONCURRENCY == 2
        assert config.TTS_MAX_CONCURRENCY == 2


class TestLLMTimeouts:
    def test_idun_post_has_timeout(self, monkeypatch):
        post = MagicMock(
            return_value=SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "hi"}}]},
                text="",
            )
        )
        monkeypatch.setattr(llm_module.requests, "post", post)
        llm_module.IdunLLM(api_key="k").generate("prompt")
        assert post.call_args.kwargs["timeout"] == Config().LLM_TIMEOUT_SECONDS

    def test_openai_client_has_timeout(self, monkeypatch):
        factory = MagicMock()
        monkeypatch.setattr(llm_module, "OpenAI", factory)
        llm_module.OpenAILLM(api_key="k")
        assert factory.call_args.kwargs["timeout"] == Config().LLM_TIMEOUT_SECONDS

    def test_gemini_client_has_timeout_in_milliseconds(self, monkeypatch):
        factory = MagicMock()
        monkeypatch.setattr(llm_module.genai, "Client", factory)
        llm_module.GeminiLLM(api_key="k")
        http_options = factory.call_args.kwargs["http_options"]
        assert http_options.timeout == Config().LLM_TIMEOUT_SECONDS * 1000


class TestEmbeddingTimeouts:
    @pytest.mark.parametrize(
        "cls", [embeddings_module.OpenAIEmbedding, embeddings_module.IdunEmbedding]
    )
    def test_openai_compatible_clients_have_timeout(self, monkeypatch, cls):
        factory = MagicMock()
        monkeypatch.setattr(embeddings_module.openai, "Client", factory)
        cls(embedding_api_key="k")
        assert factory.call_args.kwargs["timeout"] == Config().LLM_TIMEOUT_SECONDS

    def test_gemini_client_has_timeout_in_milliseconds(self, monkeypatch):
        factory = MagicMock()
        monkeypatch.setattr(embeddings_module.genai, "Client", factory)
        embeddings_module.GoogleEmbedding(embedding_api_key="k")
        http_options = factory.call_args.kwargs["http_options"]
        assert http_options.timeout == Config().LLM_TIMEOUT_SECONDS * 1000


class _ConcurrencyProbe:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def work(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.2)
        with self.lock:
            self.active -= 1


def _run_in_threads(target, count: int = 6):
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_stt_semaphore_bounds_whisper_decoding(monkeypatch):
    probe = _ConcurrencyProbe()

    def segments():
        # faster-whisper decodes lazily, while the segments are consumed.
        probe.work()
        yield SimpleNamespace(text="hi")

    model = SimpleNamespace(
        transcribe=lambda *_a, **_k: (
            segments(),
            SimpleNamespace(language="en", language_probability=1.0),
        )
    )
    monkeypatch.setattr(transcribe_module, "get_whisper_model", lambda: model)

    _run_in_threads(
        lambda: transcribe_module._transcribe_input(np.zeros(10, dtype="float32"))
    )

    assert isinstance(transcribe_module.STT_SEMAPHORE, threading.BoundedSemaphore)
    assert probe.peak == Config().STT_MAX_CONCURRENCY


def test_tts_semaphore_bounds_piper_synthesis(monkeypatch):
    probe = _ConcurrencyProbe()

    class FakeVoice:
        def synthesize_wav(self, text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            probe.work()

    service = tts_module.PiperTextToSpeechService(Config())
    monkeypatch.setattr(service, "_load_voice", lambda _name: FakeVoice())

    _run_in_threads(lambda: service.synthesize("hello", "en"))

    assert isinstance(tts_module.TTS_SEMAPHORE, threading.BoundedSemaphore)
    assert probe.peak == Config().TTS_MAX_CONCURRENCY
