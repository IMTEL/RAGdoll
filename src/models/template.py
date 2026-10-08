"""Templates that a user saves and reuses when configuring agents."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class Template(BaseModel):
    """A stored template. Only its owner can read, change or delete it.

    RAGdoll does not interpret `content`: each client decides its shape, and uses
    `type` to tell its own templates apart from other clients' templates.

    Attributes:
        id: Unique identifier, set by the database
        owner_id: The id of the user who owns the template
        type: The kind of template, chosen by the client (e.g. "interviewer")
        name: Short name shown to the user
        description: One line about the template
        content: The template itself, as the client wants it stored
        created_at: When the template was created
        updated_at: When the template was last changed
    """

    id: str | None = Field(default=None, description="Unique identifier")
    owner_id: str
    type: str
    name: str
    description: str = ""
    content: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
