from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.config import Config
from src.llm import get_models
from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.models.model import Model
from src.rag_service.dao import get_agent_dao


config = Config()
router = APIRouter()
# Repository for agent storage
access_service = access_service_factory(
    AccessServiceConfig(config.ACCESS_SERVICE, get_agent_dao())
)


# Create a new agent
@router.post("/agents/", response_model=Agent)
def create_agent(agent: Agent):
    """Create a new agent configuration.

    Args:
        agent (Agent): The agent configuration to create

    Returns:
        Agent: The created agent
    """
    return get_agent_dao().add_agent(agent)


# Get all agents
@router.get("/agents/", response_model=list[Agent])
def get_agents():
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    return get_agent_dao().get_agents()


# Get a specific agent by ID
@router.get("/agents/{agent_id}", response_model=Agent)
def get_agent(agent_id: str):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent

    Returns:
        Agent: The requested agent

    Raises:
        HTTPException: If agent not found
    """
    agent = get_agent_dao().get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    return agent


# TODO : implement a better system of returning status codes on exceptions


@router.get("/new-accesskey", response_model=AccessKey)
def new_access_key(name: str, expiery_date: datetime, agent_id: str):
    try:
        return access_service.generate_accesskey(name, expiery_date, agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/revoke-accesskey")
def revoke_access_key(access_key_id: str, agent_id: str):
    try:
        return access_service.revoke_key(access_key_id, agent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/get-accesskeys", response_model=list[AccessKey])
def get_access_keys(agent_id: str):
    agent = get_agent_dao().get_agent_by_id(agent_id)
    if agent is None:
        HTTPException(status_code=404, detail=f" agent of id not found {agent_id}")
    return agent.access_key
