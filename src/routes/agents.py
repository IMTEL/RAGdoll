from fastapi import APIRouter, HTTPException

from src.llm import get_embedding_models, get_models
from src.models.agent import Agent
from src.models.model import Model
from src.rag_service.dao import get_agent_dao


router = APIRouter()


# Create a new agent
@router.post("/agents/", response_model=Agent)
def create_agent(agent: Agent):
    """Create a new agent configuration.

    Args:
        agent (Agent): The agent configuration to create or update

    Returns:
        Agent: The created agent or the updated agent
    """
    try:
        return get_agent_dao().add_agent(agent)
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail="Invalid agentID, needs to be empty for new agents or an existing ID for updates",
        ) from e


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


@router.get("/get_models", response_model=list[Model])
def fetch_models():
    """Returns all usable models."""
    return get_models()


@router.get("/get_embedding_models", response_model=list[Model])
def fetch_embedding_models():
    """Returns all usable embedding models."""
    return get_embedding_models()
