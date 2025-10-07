from dataclasses import dataclass
from typing import Protocol

import google.generativeai as genai
import requests
from openai import OpenAI

from src.config import Config


@dataclass
class Model:
    provider: str
    name: str
    GDPR_compliant: bool | None
    description: str | None


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        """Send the prompt to the LLM and return the generated response."""

    @staticmethod
    def get_models() -> list[str]:
        """Returns the models which can be used by the provider."""


class IdunLLM(LLM):
    def __init__(self):
        self.model = Config().IDUN_MODEL
        self.url = Config().IDUN_API_URL
        self.token = Config().IDUN_API_KEY

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        response = requests.post(self.url, headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def get_models() -> list[str]:
        models = Config().IDUN_MODELS
        return [Model("idun", name, True, None) for name in models]


class OpenAILLM(LLM):
    def __init__(self):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = self.config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = OpenAI(api_key=self.config.API_KEY)

    def generate(self, prompt: str) -> str:
        """Uses the new API client interface to generate a response."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def get_models() -> list[str]:
        client = OpenAI()
        models = client.models.list()
        return [Model("openai", item.id, False, None) for item in models.data]


class GeminiLLM(LLM):
    def __init__(self):
        """Initializes the LLM facade using the provided configuration."""
        self.config = Config()
        self.model = self.config.GEMINI_MODEL

        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> str:
        """Uses the Google Generative AI client to generate a response."""
        response = self.client.generate_content(prompt)
        return response.text.strip()

    @staticmethod
    def get_models() -> list[str]:
        models = genai.list_models()
        print(models)
        return [Model("gemini", item.name, False, None) for item in models]


class MockLLM(LLM):
    def generate(self, prompt: str) -> str:
        return f"Mocked response for prompt: {prompt}"


def get_models():
    models = GeminiLLM.get_models() + IdunLLM.get_models()

    # Openai provides no Api for getting only language models
    # Gemini may also provide
    # We filter for gpt models which does not contain the selected keywords
    EXCLUDE_KEYWORDS = Config().MODEL_FILTER
    language_models = [
        model_name
        for model_name in models
        if model_name.startswith("gpt-")
        and not any(kw in model_name for kw in EXCLUDE_KEYWORDS)
    ]

    return language_models + IdunLLM.get_models()


def create_llm(llm: str = "idun") -> LLM:
    """Factory for creating LLM instances.

    Args:
        llm (str, optional): Select LLM. Defaults to "openai".

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    match llm.lower():
        case "idun":
            return IdunLLM()
        case "openai":
            return OpenAILLM()
        case "gemini":
            return GeminiLLM()
        case "mock":
            return MockLLM()
        case _:
            raise ValueError(f"LLM {llm} not supported")


if __name__ == "__main__":
    llm = create_llm("idun")
    while True:
        prompt = input("Enter your prompt: ")
        response = llm.generate(prompt)
        print(response)
