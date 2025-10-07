from datetime import datetime
from datetime import datetime
from fastapi import APIRouter, HTTPException
from typing import List

from src.auth_service.access_service import AccessServiceConfig, access_service_factory
from src.config import Config
from src.models.accesstoken import AccessKey

from src.auth_service.access_service import AccessServiceConfig, access_service_factory
from src.config import Config
from src.models.accesstoken import AccessKey
from ..models.agent import Agent, AgentRead
from ..rag_service.agent_dao import get_agent_database
from bson import ObjectId

config = Config()
config = Config()
router = APIRouter()
agent_db = get_agent_database()  # your DAO
access_service = access_service_factory(AccessServiceConfig(config.ACCESS_SERVICE,agent_db))
access_service = access_service_factory(AccessServiceConfig(config.ACCESS_SERVICE,agent_db))


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

@router.post("/agents/new-accesskey", response_model=AccessKey)
def new_access_key(name: str, expiery_date : datetime, agent_id : str):
    try:
        return access_service.generate_accesstoken(name,expiery_date,agent_id)
    except e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")


@router.get("/agents/revoke-accesskey")
def revoke_access_key(access_key: str):
    try:
        return access_service.revoke_key(access_key)
    except e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error : {e}")
    

