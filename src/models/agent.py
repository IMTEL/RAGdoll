from pydantic import BaseModel

from src.models.accesskey import AccessKey
from src.models.role import Role


class Agent(BaseModel):
    """Agent model for creating/updating agents (without ID)."""

    name: str
    description: str
    prompt: str
    corpa: list[str]
    roles: list[Role]
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_api_key: str
    access_key: list[AccessKey]  # str for now, but may need to change later
    retrieval_method: str
    embedding_model: str  # TODO: will this be in agent?
    status: str
    response_format: str
    last_updated: str
