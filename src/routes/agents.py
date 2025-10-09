from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.auth_service.access_service import AccessServiceConfig, access_service_factory
from src.config import Config
from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.rag_service.repositories import get_agent_repository


config = Config()
router = APIRouter()
agent_db = get_agent_repository()  # Repository for agent storage
access_service = access_service_factory(
    AccessServiceConfig(config.ACCESS_SERVICE, agent_db)
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
    return agent_db.create_agent(agent)


# Get all agents
@router.get("/agents/", response_model=list[Agent])
def get_agents():
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    return agent_db.get_agents()


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
    agent = agent_db.get_agent_by_id(agent_id)
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
    agent = agent_db.get_agent_by_id(agent_id)
    if agent is None:
        HTTPException(status_code=404, detail=f" agent of id not found {agent_id}")
    return agent.access_key
