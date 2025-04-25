import pytest
import time
import pytest
from uuid import uuid4
import sys
import os
from typing import Any
from src.config import Config
from src.rag_service.embeddings import create_embeddings_model, GoogleEmbedding, similarity_search
from src.LLM import OpenAI_LLM, create_llm


# === Test LLM ===

class FakeChoice:
    def __init__(self, content: str):
        # Create a fake message object with a 'content' attribute.
        self.message = type("FakeMessage", (), {"content": content})

class FakeResponse:
    def __init__(self, content: str):
        # Simulate a response with a list of choices.
        self.choices = [FakeChoice(content)]


def fake_completions_create(*, model: str, messages: Any) -> FakeResponse:
    """
    Fake function to simulate OpenAI API call.
    It ignores the inputs and returns a fake response.
    """
    return FakeResponse("Test response")



@pytest.mark.unit
def test_create_prompt():
    """
    Test that create_prompt correctly appends additional context.
    """
    # Instantiate an LLM instance. (No API call is made here.)
    llm = OpenAI_LLM()
    base_prompt = "Explain the significance of Python."
    # Provide some extra context as keyword arguments.
    prompt = llm.create_prompt(base_prompt, audience="beginner", detail="basic overview")
    expected_prompt = "Explain the significance of Python.\naudience: beginner\ndetail: basic overview"
    assert prompt == expected_prompt, "create_prompt should combine the base prompt with additional context"


@pytest.fixture
def patched_llm(monkeypatch):
    """
    Fixture to return an OpenAI_LLM instance with the API call patched to avoid spending tokens.
    """
    llm = OpenAI_LLM()
    # Patch the chat.completions.create method so that it returns our fake response.
    monkeypatch.setattr(llm.client.chat.completions, "create", fake_completions_create)
    return llm


@pytest.mark.unit
def test_generate_with_patched_llm(patched_llm):
    """
    Test that generate returns the fake response from the patched API call.
    """
    test_prompt = "Dummy prompt"
    response = patched_llm.generate(test_prompt)
    assert response == "Test response", "generate should return the fake response content"


@pytest.mark.unit
def test_create_llm_valid():
    """
    Test that create_llm returns an instance of OpenAI_LLM when provided 'openai'.
    """
    llm_instance = create_llm("openai")
    assert isinstance(llm_instance, OpenAI_LLM), "create_llm('openai') should return an OpenAI_LLM instance"


@pytest.mark.unit
def test_create_llm_invalid():
    """
    Test that create_llm raises a ValueError when an unsupported LLM is specified.
    """
    with pytest.raises(ValueError) as exc_info:
        create_llm("unsupported")
    assert "not supported" in str(exc_info.value), "create_llm should raise ValueError for unsupported LLMs"


@pytest.mark.integration
def test_llm_comparison():
    """
    Test that compares responses from both OpenAI and Gemini models.
    Demonstrates how to use both models and display their outputs.
    """
    # Test prompt
    base_prompt = "What are the key differences between Python and JavaScript?"
    
    # Create instances of both LLM types
    openai_llm = create_llm("openai")
    gemini_llm = create_llm("gemini")
    
    # Generate prompts with the same context
    context = {"audience": "beginners", "max_length": "brief"}
    openai_prompt = openai_llm.create_prompt(base_prompt, **context)
    gemini_prompt = gemini_llm.create_prompt(base_prompt, **context)
    
    # Generate responses
    print("\n========== GENERATING RESPONSES ==========")
    
    print("\n[OPENAI MODEL]:", openai_llm.model)
    openai_response = openai_llm.generate(openai_prompt)
    print(openai_response)
    
    print("\n[GEMINI MODEL]:", gemini_llm.model)
    gemini_response = gemini_llm.generate(gemini_prompt)
    print(gemini_response)
    
    print("\n========== END OF RESPONSES ==========\n")
    
    # Basic assertions to verify responses were generated
    assert len(openai_response) > 0, "OpenAI response should not be empty"
    assert len(gemini_response) > 0, "Gemini response should not be empty"
    assert openai_response != gemini_response, "Responses should differ between models"
    
    
import asyncio
from types import SimpleNamespace

import pytest

from src.LLM import OpenAI_LLM
import src.streaming_ws as streaming_ws

# --------------------------------------------------------------------------- #
# Helper fakes                                                                
# --------------------------------------------------------------------------- #
from collections import deque
from types import SimpleNamespace

class _FakeDelta(SimpleNamespace):      # exposes .content
    pass

class _FakeChunk(SimpleNamespace):      # exposes .choices[0].delta.content
    def __init__(self, token: str):
        super().__init__(choices=[SimpleNamespace(delta=_FakeDelta(content=token))])

class _FakeStream:
    """Mimic what OpenAI returns when stream=True."""
    def __init__(self, tokens):
        self._tokens = deque(tokens)

    # -- awaitable ---------------------------------------------------------- #
    def __await__(self):
        async def _dummy():              # so `await _FakeStream(...)` works
            return self
        return _dummy().__await__()

    # -- async-iterable ----------------------------------------------------- #
    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._tokens:
            raise StopAsyncIteration
        return _FakeChunk(self._tokens.popleft())

async def _fake_async_create(**kwargs):
    return _FakeStream(["Hello", " ", "world", "!"])


class _MockStreamingLLM:
    """Lean stand-in used to isolate `stream_chat_completion`."""
    def __init__(self):
        self.prompt_seen = None

    async def astream(self, prompt: str):
        self.prompt_seen = prompt
        for tok in ["foo", "bar"]:
            yield tok


# --------------------------------------------------------------------------- #
# Helper fakes                                                                
# --------------------------------------------------------------------------- #









# --------------------------------------------------------------------------- #
# 1.  OpenAI_LLM.astream                                                      
# --------------------------------------------------------------------------- #

pytestmark = [pytest.mark.unit]   # applies to every test in this file


@pytest.mark.asyncio
async def test_openai_astream(monkeypatch):
    """
    Patched `OpenAI_LLM.astream` should emit every token coming from the fake
    async client without blocking the event-loop.
    """
    llm = OpenAI_LLM()

    # Patch ONLY the network call – the rest of the logic stays intact
    monkeypatch.setattr(
    llm.client.chat.completions,
    "create",
    _fake_async_create,
    raising=True,
)


    tokens = []
    async for tok in llm.astream("dummy prompt"):
        tokens.append(tok)

    assert "".join(tokens) == "Hello world!"


# --------------------------------------------------------------------------- #
# 2.  streaming_ws.stream_chat_completion                                      
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_stream_chat_completion_uses_astream(monkeypatch):
    """
    Verify that `stream_chat_completion` delegates to LLM.astream and yields the
    *exact* stream it produces.
    """

    mock_llm = _MockStreamingLLM()

    # Monkey-patch the factory that streaming_ws relies on
    monkeypatch.setattr(streaming_ws, "create_llm", lambda _: mock_llm, raising=True)

    tokens = []
    async for tok in streaming_ws.stream_chat_completion("whatever", model="openai"):
        tokens.append(tok)

    assert tokens == ["foo", "bar"]
    assert mock_llm.prompt_seen == "whatever", "prompt should be forwarded unchanged"

