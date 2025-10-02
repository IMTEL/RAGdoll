from pydantic import BaseModel

from src.models.accesskey import AccessKey
from .role import Role

class Agent(BaseModel):
    name: str
    description: str
    prompt: str
    corpa: list[str]
    roles: list[Role]
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_api_key: str
    access_key: list[AccessKey]
    retrieval_method: str
    embedding_model: str # idk if will be in agent
    status: str
    response_format: str
    lastUpdated: str

class AgentRead(BaseModel):
    id: str
    name: str
    description: str
    prompt: str
    corpa: list[str]
    roles: list[Role]
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_api_key: str
    access_key: list[AccessKey] 
    retrieval_method: str
    embedding_model: str  # TODO: will this be in agent?
    status: str
    response_format: str
    last_updated: str
