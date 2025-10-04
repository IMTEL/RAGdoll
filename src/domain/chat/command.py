"""Command and request models for the chat service.

This module defines the data structures for incoming requests from the VR application
and the prompts sent to language models.
"""

from pydantic import BaseModel, Field, ValidationError

from src.domain.chat import Message
from src.domain.training import ProgressData


class Command(BaseModel):
    """Request from the VR application containing user state and agent configuration.

    This command encapsulates all information needed to generate an AI response:
    - Which AI agent to use
    - Which roles within that agent are active (for RAG access control)
    - User's progress and actions
    - Conversation history

    Attributes:
        chat_log: Conversation history
        agent_id: ID of the agent to use for generating response
        active_role_ids: List of role names that determine corpus access
        access_key: API key for authentication (optional)
    """

    chat_log: list[Message] = Field(default_factory=list)

    # Agent configuration
    agent_id: str = Field(..., description="MongoDB ObjectId of the agent to use")
    active_role_ids: list[str] = Field(
        default_factory=list,
        description="Names of roles that determine RAG corpus access",
    )
    access_key: str | None = Field(
        default=None, description="API key for agent access authorization"
    )


class Prompt(BaseModel):
    """Structured prompt to be sent to a language model.

    This is the final prompt assembled from the Command after:
    - Retrieving relevant context from RAG
    - Combining agent's system prompt
    - Formatting conversation history

    Attributes:
        user_information: User context strings
        question: The current user question
        progress: User's training progress
        user_actions: Recent user actions
        base_prompt: The agent's system prompt
        context: Retrieved RAG context relevant to the question
    """

    user_information: list[str] = Field(default_factory=list)
    question: str
    progress: list[ProgressData] = Field(default_factory=list)
    user_actions: list[str] = Field(default_factory=list)
    base_prompt: str
    context: str


def prompt_to_json(prompt: Prompt) -> str:
    """Converts a Prompt instance to a JSON string."""
    return prompt.model_dump_json()


def command_from_json(json_str: str) -> Command | None:
    """Parses a JSON string into a Command object.

    Args:
        json_str (str): The JSON string representing a Command.

    Returns:
        Optional[Command]: The Command object if parsing is successful, otherwise None.
    """
    try:
        command = Command.model_validate_json(json_str)
        return command
    except ValidationError as e:
        print("Validation error:", e)
        return None


def command_from_json_transcribe_version(
    json_str: str, question: str | None = None
) -> Command | None:
    """Parses a JSON string into a Command object.

    Args:
        json_str (str): The JSON string representing a Command.
        question (str | None): An optional question to append to the chat log.

    Returns:
        Optional[Command]: The Command object if parsing is successful, otherwise None.
    """
    try:
        command = Command.model_validate_json(json_str)

        if question:
            command.chat_log.append(Message(role="user", content=question))
        else:
            command.chat_log.append(Message(role="user", content=""))
        return command
    except ValidationError as e:
        print("Validation error:", e)
        return None
