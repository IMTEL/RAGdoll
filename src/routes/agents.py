from fastapi import APIRouter, HTTPException

from src.models.agent import Agent
from src.rag_service.dao import get_agent_dao


router = APIRouter()
agent_db = get_agent_dao()  # DAO for agent storage


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
