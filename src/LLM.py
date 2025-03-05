from typing import Protocol
import openai
from config import Config  # assuming config.py is in your project

class LLMFacadeProtocol(Protocol):
    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Combines a base prompt with additional context to form a complete prompt.
        :param base_prompt: The main prompt text.
        :param kwargs: Additional key-value context to include.
        :return: A string representing the full prompt.
        """
        ...

    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to the LLM and returns the generated response.
        :param prompt: The complete prompt for the LLM.
        :return: The LLM's response.
        """
        ...


class LLMFacade(LLMFacadeProtocol):
    def __init__(self, config: Config):
        """
        Initializes the LLM facade using the provided configuration.
        :param config: An instance of Config containing API keys and model details.
        """
        self.config = config
        openai.api_key = config.API_KEY
        self.model = config.GPT_MODEL

    def create_prompt(self, base_prompt: str, **kwargs) -> str:
        """
        Creates a prompt by appending additional context to the base prompt.
        :param base_prompt: The main prompt text.
        :param kwargs: Additional key-value context to include.
        :return: The combined prompt.
        """
        # Combine additional context into a single string.
        additional_context = "\n".join(f"{key}: {value}" for key, value in kwargs.items())
        return f"{base_prompt}\n{additional_context}" if additional_context else base_prompt

    def generate(self, prompt: str) -> str:
        """
        Sends the prompt to the LLM using OpenAI's ChatCompletion API and returns the response.
        :param prompt: The complete prompt for the LLM.
        :return: The generated text response.
        """
        # Prepare the messages list for a chat-based model.
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        # Call the API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        # Extract and return the assistant's response.
        return response.choices[0].message.content.strip()


# Example usage:
if __name__ == "__main__":
    # Initialize configuration and facade.
    config = Config()
    llm = LLMFacade(config)
    
    # Create a full prompt.
    base_prompt = "Explain the significance of the Python programming language."
    full_prompt = llm.create_prompt(base_prompt, audience="beginner", detail="basic overview")
    
    # Generate a response from the LLM.
    response = llm.generate(full_prompt)
    print("LLM Response:")
    print(response)




