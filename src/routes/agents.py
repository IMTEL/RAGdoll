from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi_jwt_auth import AuthJWT

from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.auth.auth_service.factory import auth_service_factory
from src.config import Config
from src.llm import get_models
from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.models.model import Model
from src.rag_service.dao import get_agent_dao
from src.rag_service.dao.factory import get_user_dao
from src.rag_service.embeddings import get_available_embedding_models


config = Config()
router = APIRouter()


user_dao = get_user_dao()
agent_dao = get_agent_dao()

access_service = access_service_factory(
    AccessServiceConfig(config.ACCESS_SERVICE, get_agent_dao())
)
auth_service = auth_service_factory(config.AUTH_SERVICE)


# Update agent
@router.post("/update-agent/", response_model=Agent)
def create_agent(agent: Agent, authorize: Annotated[AuthJWT, Depends()] = None):
    """Create a new agent configuration.

    Args:
        agent (Agent): The agent configuration to create
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        Agent: The created agent
    """
    try:
        # Check if agent exists
        if agent_dao.get_agent_by_id(agent.id) is None:
            # Authenticate and get user
            user = auth_service.get_authenticated_user(authorize)
            agent = get_agent_dao().add_agent(agent)
            # Add new agent id to owned agents
            user.owned_agents.append(agent.id)
            user_dao.set_user(user)
            return agent

        auth_service.auth(authorize, agent.id)
        return get_agent_dao().add_agent(agent)

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail="Invalid agentID, needs to be empty for new agents or an existing ID for updates",
        ) from e


# Get all agents
@router.get("/agents/", response_model=list[Agent])
def get_agents(authorize: Annotated[AuthJWT, Depends()] = None):
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    user = auth_service.get_authenticated_user(authorize)

    # returns all agents, owned by the user
    return [get_agent_dao().get_agent_by_id(agent_id) for agent_id in user.owned_agents]


# Get a specific agent by ID
@router.get("/fetch-agent", response_model=Agent)
def get_agent(agent_id: str, authorize: Annotated[AuthJWT, Depends()] = None):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        Agent: The requested agent

    Raises:
        HTTPException: If agent not found
    """
    auth_service.auth(authorize, agent_id)
    agent = get_agent_dao().get_agent_by_id(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    return agent


# Get a specific agent by ID using AccessKey for authentication
@router.get("/agent-info", response_model=Agent)
def agent_info(
    agent_id: str,
    access_key: Annotated[str | None, Header()],
):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        access_key (Annotated[str | None, Header()]): access key header

    Returns:
        Agent: The requested agent

    Raises:
        HTTPException: If agent not found
    """
    if not access_service.authenticate(agent_id, access_key):
        raise HTTPException(
            status_code=401, detail="Access key not valid for agent, Unauthorized"
        )

    agent = get_agent_dao().get_agent_by_id(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    return agent


# TODO : implement a better system of returning status codes on exceptions


@router.get("/new-accesskey", response_model=AccessKey)
def new_access_key(
    name: str,
    agent_id: str,
    expiry_date: str | None = None,
    authorize: Annotated[AuthJWT, Depends()] = None,
):
    try:
        auth_service.auth(authorize, agent_id)
        if expiry_date is None:
            return access_service.generate_accesskey(name, None, agent_id)
        else:
            expiry_date_formatted = datetime.fromisoformat(expiry_date).replace(
                tzinfo=None
            )
            return access_service.generate_accesskey(
                name, expiry_date_formatted, agent_id
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/revoke-accesskey")
def revoke_access_key(
    access_key_id: str, agent_id: str, authorize: Annotated[AuthJWT, Depends()] = None
):
    auth_service.auth(authorize, agent_id)
    try:
        return access_service.revoke_key(agent_id, access_key_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/get-accesskeys", response_model=list[AccessKey])
def get_access_keys(agent_id: str, authorize: Annotated[AuthJWT, Depends()] = None):
    auth_service.auth(authorize, agent_id)
    agent = get_agent_dao().get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f" agent of id not found {agent_id}"
        )
    access_keys = agent.access_key
    for access_key in access_keys:
        access_key.key = None
    return access_keys


@router.get("/get_models", response_model=list[Model])
def fetch_models(authorize: Annotated[AuthJWT, Depends()] = None):
    """Returns all usable models."""
    authorize.jwt_required()  # Require login, but nothing else
    return get_models()


@router.get("/get_embedding_models", response_model=list[str])
def fetch_embedding_models():
    """Returns all usable embedding models."""
    return get_available_embedding_models()
