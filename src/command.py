from pydantic import BaseModel
from typing import Optional



class Command(BaseModel):
    """Message from the VR application about the current state. This is an loose implementation of the command pattern
    """
    user_name: str
    user_mode: str
    question: str
    progress: list[str]
    user_actions: list[str]
    NPC: int


class Prompt(BaseModel):
    """Message to be passed to a large language model."""
    user_name: str
    user_mode: str
    question: str
    progress: list[str]
    user_actions: list[str]
    base_prompt: str
    context: str


def prompt_to_json(prompt: Prompt) -> str:
    """Converts a Prompt instance to a JSON string."""
    return prompt.model_dump_json()