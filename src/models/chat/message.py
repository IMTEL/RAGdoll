"""Chat domain models for message handling."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation.

    Represents one turn in a chat conversation, containing
    the role (system/user/assistant) and content.

    Attributes:
        role: The message sender role ("system", "user", "assistant")
        content: The actual message text
    """

    role: str = Field(..., description="Message sender role")
    content: str = Field(..., description="Message content")
