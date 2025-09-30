from typing import Protocol
from openai import OpenAI
import requests
import google.generativeai as genai
from src.config import Config 
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider : str
    api_key: str
    model: str


class LLM(Protocol):

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to the LLM and return the generated response.
        """
        pass


class Idun_LLM(LLM):
    def __init__(self, model : str, api_key : str):
        self.model = model
        self.api_key = api_key
        self.url = Config().IDUN_API_URL

    def generate(self, prompt: str) -> str:
        
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        data = {
        "model": self.model,
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ]
        }
        response = requests.post(self.url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()

class OpenAI_LLM(LLM):
    def __init__(self, model : str, api_key : str):
        """
        Initializes the LLM facade using the provided configuration.
        """
        self.model = model
        self.api_key = api_key
        # Instantiate the client using the new OpenAI interface.
        self.client = OpenAI(api_key=self.config.API_KEY)

    def generate(self, prompt: str) -> str:
        """
        Uses the new API client interface to generate a response.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content.strip()


class Gemini_LLM(LLM):
    def __init__(self, model : str, api_key : str):
        """
        Initializes the LLM facade using the provided configuration.
        """
        self.api_key = api_key
        self.model = self.model

        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> str:
        """
        Uses the Google Generative AI client to generate a response
        """
        response = self.client.generate_content(prompt)
        return response.text.strip()
    
class MockLLM(LLM):

    def __init__(self, model: str, api_key: str) -> None:
        pass
    
    def generate(self, prompt: str) -> str:
        return f"Mocked response for prompt: {prompt}"


def llm_factory(config : LLMConfig) -> LLM:
    """
    Factory for creating LLM instances.

    Args:
        llm (str, optional): Select LLM. Defaults to "openai".

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    match config.provider.lower():
        case "idun":
            return Idun_LLM(config.model, config.api_key)
        case "openai":
            return OpenAI_LLM(config.model, config.api_key)
        case "gemini":
            return Gemini_LLM(config.model, config.api_key)
        case "mock":
            return MockLLM(config.model, config.api_key)
        case _:
            raise ValueError(f"LLM {config.provider} not supported")
        
        
if __name__ == "__main__":
    llm = llm_factory("idun")
    while True:
        prompt = input("Enter your prompt: ")
        response = llm.generate(prompt)
        print(response)
    