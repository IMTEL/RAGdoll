from typing import Protocol

import google.generativeai as genai
import requests
from openai import OpenAI

from src.config import Config


class LLM(Protocol):
    def generate(self, prompt: str) -> str:
        """Send the prompt to the LLM and return the generated response."""


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

        # Handle potential errors
        if response.status_code != 200:
            raise Exception(
                f"Error from Idun API: {response.status_code} - {response.text}"
            )

        return response.json()["choices"][0]["message"]["content"].strip()


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


class MockLLM(LLM):
    def generate(self, prompt: str) -> str:
        return f"Mocked response for prompt: {prompt}"


def create_llm(llm_provider: str = "idun") -> LLM:
    """Factory for creating LLM instances.

    Args:
        llm_provider (str): The LLM service provider ("idun", "openai", "gemini", "mock").

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    """
    TODO: Add support for:
    - Choosing model variants (e.g., GPT-4-turbo)
    - API key management
    - Advanced parameters (temperature, max tokens, etc.)
    """
    match llm_provider.lower():
        case "idun":
            return IdunLLM()
        case "openai":
            return OpenAILLM()
        case "gemini":
            return GeminiLLM()
        case "mock":
            return MockLLM()
        case _:
            raise ValueError(f"LLM {llm_provider} not supported")


if __name__ == "__main__":
    llm = create_llm("idun")
    while True:
        prompt = input("Enter your prompt: ")
        response = llm.generate(prompt)
        print(response)
