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
    prompt = base_prompt
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

