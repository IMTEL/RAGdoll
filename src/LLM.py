from typing import Protocol
from openai import OpenAI
import google.generativeai as genai
from src.config import Config 

class LLM(Protocol):
    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Combine a base prompt with additional context.
        """
        pass

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to the LLM and return the generated response.
        """
        pass

# OPENAI

class OpenAI_LLM(LLM):
    def __init__(self):
        """
        Initializes the LLM facade using the provided configuration.
        """
        self.config = Config()
        self.model = self.config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = OpenAI(api_key=self.config.API_KEY)

    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Creates a prompt by appending additional context.
        """
        additional_context = "\n".join(f"{key}: {value}" for key, value in kwargs.items())
        return f"{base_prompt}\n{additional_context}" if additional_context else base_prompt

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

# GOOGLE GEMINI

class Gemini_LLM(LLM):
    def __init__(self):
        """
        Initializes the LLM facade using the provided configuration.
        """
        self.config = Config()
        self.model = self.config.GEMINI_MODEL

        genai.configure(api_key=self.config.GEMINI_API_KEY)
        self.client = genai.GenerativeModel(self.model)

    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Creates a prompt by appending additional context.
        """
        additional_context = "\n".join(f"{key}: {value}" for key, value in kwargs.items())
        return f"{base_prompt}\n{additional_context}" if additional_context else base_prompt

    def generate(self, prompt: str) -> str:
        """
        Uses the Google Generative AI client to generate a response
        """
        response = self.client.generate_content(prompt)
        return response.text.strip()
    
class MockLLM(LLM):
    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        return f"Mocked prompt for base prompt: {base_prompt}"
    
    def generate(self, prompt: str) -> str:
        return f"Mocked response for prompt: {prompt}"
    
    
def create_llm(llm: str = "openai") -> LLM:
    """
    Factory for creating LLM instances.

    Args:
        llm (str, optional): Select LLM. Defaults to "openai".

    Raises:
        ValueError: If the specified LLM is not supported.

    Returns:
        LLM: The specified LLM instance.
    """
    match llm.lower():
        case "openai":
            return OpenAI_LLM()
        case "gemini":
            return Gemini_LLM()
        case "mock":
            return MockLLM()
        case _:
            raise ValueError(f"LLM {llm} not supported")