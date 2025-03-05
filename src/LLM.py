from typing import Protocol
from openai import OpenAI
from config import Config

class LLMFacadeProtocol(Protocol):
    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Combine a base prompt with additional context.
        """
        ...

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to the LLM and return the generated response.
        """
        ...

class LLMFacade(LLMFacadeProtocol):
    def __init__(self, config: Config):
        """
        Initializes the LLM facade using the provided configuration.
        """
        self.config = config
        self.model = config.GPT_MODEL
        # Instantiate the client using the new OpenAI interface.
        self.client = OpenAI(api_key=config.API_KEY)

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


# Example usage:
if __name__ == "__main__":
    config = Config()
    llm = LLMFacade(config)
    
    base_prompt = "Explain the significance of the Python programming language."
    full_prompt = llm.create_prompt(base_prompt, audience="beginner", detail="basic overview")
    
    response = llm.generate(full_prompt)
    print("LLM Response:")
    print(response)
