import os
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from src.config import Config
from src.globals import access_service, agent_dao, auth_service, user_dao
from src.llm import list_llm_models
from src.models.accesskey import AccessKey
from src.models.agent import Agent
from src.models.errors.embedding_error import EmbeddingAPIError, EmbeddingError
from src.models.errors.llm_error import LLMAPIError
from src.models.model import Model
from src.models.users.user import User
from src.rag_service.embeddings import list_embedding_models


config = Config()
router = APIRouter()


def optional_auth(request: Request) -> Optional[AuthJWT]:
    """Return AuthJWT only if auth is enabled and header is present."""
    if os.getenv("DISABLE_AUTH", "").lower() == "true":
        return None

    auth_header = request.headers.get("authorization")
    if not auth_header:
        # If no header and auth is required, AuthJWT will handle the error
        pass

    return AuthJWT(request)


def _get_user_or_demo(authorize: Optional[AuthJWT]) -> User:
    """Get authenticated user or demo user if auth is disabled."""
    if os.getenv("DISABLE_AUTH", "").lower() == "true" or authorize is None:
        demo_user = user_dao.get_user_by_email("demo@example.com")
        if not demo_user:
            demo_user = User(
                email="demo@example.com",
                name="Demo User",
                api_keys=[],
                owned_agents=[],
                auth_provider="demo",
                provider_user_id="demo",
            )
            user_dao.set_user(demo_user)
        return demo_user
    return auth_service.get_authenticated_user(authorize)


def _auth_or_skip(authorize: Optional[AuthJWT], agent_id: str):
    """Check agent ownership or skip if auth is disabled."""
    if os.getenv("DISABLE_AUTH", "").lower() == "true" or authorize is None:
        return
    auth_service.auth(authorize, agent_id)


class ProviderKeyRequest(BaseModel):
    provider: str
    api_key: str


def _map_embedding_api_error(error: EmbeddingAPIError) -> int:
    message = str(error.original_error).lower() if error.original_error else ""
    if any(keyword in message for keyword in ("quota", "rate limit", "429")):
        return 429
    if any(
        keyword in message
        for keyword in (
            "unauthorized",
            "forbidden",
            "authentication",
            "api key",
            "permission",
        )
    ):
        return 401
    return 400


# Update agent
@router.post("/update-agent/", response_model=Agent)
def create_agent(
    agent: Agent, authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None
):
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
            user = _get_user_or_demo(authorize)  # Changed
            agent = agent_dao.add_agent(agent)
            # Add new agent id to owned agents
            user.owned_agents.append(agent.id)
            user_dao.set_user(user)
            return agent

        _auth_or_skip(authorize, agent.id)  # Changed
        return agent_dao.add_agent(agent)

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail="Invalid agentID, needs to be empty for new agents or an existing ID for updates",
        ) from e


# Get all agents
@router.get("/agents/", response_model=list[Agent])
def get_agents(authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None):
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    user = _get_user_or_demo(authorize)  # Changed

    # returns all agents, owned by the user
    return [agent_dao.get_agent_by_id(agent_id) for agent_id in user.owned_agents]


# Get a specific agent by ID
@router.get("/delete-agent")
def delete_agent(
    agent_id: str,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    """Deletes a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        HTTP respone code

    Raises:
        HTTPException: If agent not found
    """
    user = _get_user_or_demo(authorize)  # Changed
    agent_dao.delete_agent_by_id(agent_id)
    user.owned_agents.remove(agent_id)
    user_dao.set_user(user)


# Get a specific agent by ID
@router.get("/fetch-agent", response_model=Agent)
def get_agent(
    agent_id: str,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        Agent: The requested agent

    Raises:
        HTTPException: If agent not found
    """
    _auth_or_skip(authorize, agent_id)  # Changed
    agent = agent_dao.get_agent_by_id(agent_id)

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

    agent = agent_dao.get_agent_by_id(agent_id)

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
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    try:
        _auth_or_skip(authorize, agent_id)  # Changed
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
    access_key_id: str,
    agent_id: str,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    _auth_or_skip(authorize, agent_id)  # Changed
    try:
        return access_service.revoke_key(agent_id, access_key_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/get-accesskeys", response_model=list[AccessKey])
def get_access_keys(
    agent_id: str,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    _auth_or_skip(authorize, agent_id)  # Changed
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f" agent of id not found {agent_id}"
        )
    access_keys = agent.access_key
    for access_key in access_keys:
        access_key.key = None
    return access_keys


@router.post("/get_models", response_model=list[Model])
def fetch_models(
    payload: ProviderKeyRequest,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    """Return all usable models for the requested provider using the supplied API key."""
    if os.getenv("DISABLE_AUTH", "").lower() != "true":
        authorize.jwt_required()  # Only require JWT if auth is enabled

    try:
        return list_llm_models(payload.provider, payload.api_key)
    except LLMAPIError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/get_embedding_models", response_model=list[str])
def fetch_embedding_models(
    payload: ProviderKeyRequest,
    authorize: Annotated[Optional[AuthJWT], Depends(optional_auth)] = None,
):
    """Return all usable embedding models for the requested provider using the supplied API key."""
    if os.getenv("DISABLE_AUTH", "").lower() != "true":
        authorize.jwt_required()  # Only require JWT if auth is enabled

    try:
        return list_embedding_models(payload.provider, payload.api_key)
    except EmbeddingAPIError as error:
        status_code = _map_embedding_api_error(error)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except EmbeddingError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
