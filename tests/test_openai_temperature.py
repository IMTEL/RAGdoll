"""OpenAI's reasoning models only accept their default temperature, so none is sent to them."""

from types import SimpleNamespace

import pytest

from src.llm import OpenAILLM


class _RecordingCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="Hei!")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _generate(model: str, temperature: float) -> dict:
    llm = OpenAILLM(model=model, api_key="key")
    completions = _RecordingCompletions()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm.generate("Hei", temperature=temperature)
    return completions.kwargs


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4.1-mini"])
def test_sends_the_temperature_to_ordinary_models(model):
    assert _generate(model, 0.2)["temperature"] == 0.2


@pytest.mark.parametrize("model", ["o1", "o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"])
def test_sends_no_temperature_to_reasoning_models(model):
    assert _generate(model, 0.2)["temperature"] is None
