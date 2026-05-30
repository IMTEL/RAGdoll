import os
from datetime import datetime
from typing import Annotated

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


def _auth_disabled() -> bool:
    return os.getenv("DISABLE_AUTH", "").lower() == "true" or config.RUNNING_TESTS


def optional_auth(request: Request) -> AuthJWT | None:
    """Return AuthJWT only if auth is enabled and header is present."""
    if _auth_disabled():
        return None

    auth_header = request.headers.get("authorization")
    if not auth_header:
        # If no header and auth is required, AuthJWT will handle the error
        pass

    return AuthJWT(request)


def _get_user_or_demo(authorize: AuthJWT | None) -> User:
    """Get authenticated user or demo user if auth is disabled."""
    if _auth_disabled() or authorize is None:
        demo_user = user_dao.get_user_by_provider("demo", "demo")
        if not demo_user:
            demo_user = User(
                email="demo@example.com",
                name="Demo User",
                api_keys=[],
                owned_agents=[],
                auth_provider="demo",
                provider_user_id="demo",
            )
            demo_user = user_dao.set_user(demo_user)
        return demo_user
    return auth_service.get_authenticated_user(authorize)


def _auth_or_skip(authorize: AuthJWT | None, agent_id: str):
    """Check agent ownership or skip if auth is disabled."""
    if _auth_disabled() or authorize is None:
        return
    auth_service.auth(authorize, agent_id)


def _ensure_agent_owner(authorize: AuthJWT | None, agent_id: str) -> User | None:
    if _auth_disabled() or authorize is None:
        return None

    user = auth_service.get_authenticated_user(authorize)
    if agent_id not in user.owned_agents:
        raise HTTPException(status_code=401, detail="Unauthorized edit of agent")
    return user


def _can_access_agent(user: User, agent_id: str) -> bool:
    return agent_id in user.owned_agents or agent_id in user.collaborating_agents


def _get_agent_owner(agent_id: str) -> User | None:
    for user in user_dao.get_users_with_agent(agent_id):
        if agent_id in user.owned_agents:
            return user
    return None


def _public_user(user: User, role: str | None = None) -> dict:
    return {
        "id": user.id or "",
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
        "role": role,
    }


def _scrub_agent_api_keys(agent: Agent) -> Agent:
    agent_copy = agent.model_copy()
    agent_copy.llm_api_key = ""
    agent_copy.embedding_api_key = ""
    return agent_copy


class UserSearchResult(BaseModel):
    id: str
    name: str | None = None
    email: str | None = None
    picture: str | None = None
    role: str | None = None


class CollaboratorInviteRequest(BaseModel):
    user_id: str


class CollaboratorsResponse(BaseModel):
    owner: UserSearchResult | None
    collaborators: list[UserSearchResult]
    current_user_id: str | None = None
    is_owner: bool = False


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
    agent: Agent, authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None
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
        existing_agent = agent_dao.get_agent_by_id(agent.id)
        if existing_agent is None:
            # Authenticate and get user
            user = _get_user_or_demo(authorize)  # Changed
            agent = agent_dao.add_agent(agent)
            # Add new agent id to owned agents
            user.owned_agents.append(agent.id)
            user_dao.set_user(user)
            return agent

        _auth_or_skip(authorize, agent.id)  # Changed
        user = _get_user_or_demo(authorize)
        if not agent.llm_api_key:
            agent.llm_api_key = existing_agent.llm_api_key
        if not agent.embedding_api_key:
            agent.embedding_api_key = existing_agent.embedding_api_key

        updated_agent = agent_dao.add_agent(agent)
        if agent.id not in user.owned_agents:
            return _scrub_agent_api_keys(updated_agent)
        return updated_agent

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail="Invalid agentID, needs to be empty for new agents or an existing ID for updates",
        ) from e


# Get all agents
@router.get("/agents/", response_model=list[Agent])
def get_agents(authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None):
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    user = _get_user_or_demo(authorize)  # Changed

    owned_agents = [
        agent
        for agent_id in user.owned_agents
        if (agent := agent_dao.get_agent_by_id(agent_id)) is not None
    ]
    collaborator_agents = [
        _scrub_agent_api_keys(agent)
        for agent_id in user.collaborating_agents
        if (agent := agent_dao.get_agent_by_id(agent_id)) is not None
    ]
    return owned_agents + collaborator_agents


# Get a specific agent by ID
@router.get("/delete-agent")
def delete_agent(
    agent_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
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
    _ensure_agent_owner(authorize, agent_id)
    agent_dao.delete_agent_by_id(agent_id)
    if agent_id in user.owned_agents:
        user.owned_agents.remove(agent_id)
    user_dao.set_user(user)
    for collaborator in user_dao.get_users_with_agent(agent_id):
        if agent_id in collaborator.collaborating_agents:
            collaborator.collaborating_agents.remove(agent_id)
            user_dao.set_user(collaborator)


# Get a specific agent by ID
@router.get("/fetch-agent", response_model=Agent)
def get_agent(
    agent_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
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
    user = _get_user_or_demo(authorize)
    agent = agent_dao.get_agent_by_id(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    if agent_id not in user.owned_agents:
        return _scrub_agent_api_keys(agent)
    return agent


@router.get("/users/search", response_model=list[UserSearchResult])
def search_users(
    q: str,
    limit: int = 10,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    current_user = _get_user_or_demo(authorize)
    users = user_dao.search_users(q, min(max(limit, 1), 25))
    return [
        UserSearchResult(**_public_user(user))
        for user in users
        if user.id is not None and user.id != current_user.id
    ]


@router.get(
    "/agents/{agent_id}/collaborators",
    response_model=CollaboratorsResponse,
)
def get_collaborators(
    agent_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    current_user = _get_user_or_demo(authorize)
    if not _can_access_agent(current_user, agent_id):
        raise HTTPException(status_code=401, detail="Unauthorized access to agent")

    users = user_dao.get_users_with_agent(agent_id)
    owner = next((user for user in users if agent_id in user.owned_agents), None)
    collaborators = [
        UserSearchResult(**_public_user(user, "collaborator"))
        for user in users
        if agent_id in user.collaborating_agents and user.id is not None
    ]

    return CollaboratorsResponse(
        owner=UserSearchResult(**_public_user(owner, "owner")) if owner else None,
        collaborators=collaborators,
        current_user_id=current_user.id,
        is_owner=agent_id in current_user.owned_agents,
    )


@router.post(
    "/agents/{agent_id}/collaborators",
    response_model=CollaboratorsResponse,
)
def add_collaborator(
    agent_id: str,
    payload: CollaboratorInviteRequest,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    owner = _ensure_agent_owner(authorize, agent_id)
    invited_user = user_dao.get_user_by_id(payload.user_id)
    if invited_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    if owner and invited_user.id == owner.id:
        raise HTTPException(status_code=400, detail="Owner is already on this agent")
    if agent_id in invited_user.owned_agents:
        raise HTTPException(status_code=400, detail="User already owns this agent")
    if agent_id not in invited_user.collaborating_agents:
        invited_user.collaborating_agents.append(agent_id)
        user_dao.set_user(invited_user)
    return get_collaborators(agent_id, authorize)


@router.delete(
    "/agents/{agent_id}/collaborators/{user_id}",
    response_model=CollaboratorsResponse,
)
def remove_collaborator(
    agent_id: str,
    user_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    owner = _ensure_agent_owner(authorize, agent_id)
    if owner and user_id == owner.id:
        raise HTTPException(status_code=400, detail="Owner cannot be removed")

    collaborator = user_dao.get_user_by_id(user_id)
    if collaborator is None:
        raise HTTPException(status_code=404, detail="User not found")
    if agent_id in collaborator.collaborating_agents:
        collaborator.collaborating_agents.remove(agent_id)
        user_dao.set_user(collaborator)
    return get_collaborators(agent_id, authorize)


@router.post("/agents/{agent_id}/leave")
def leave_agent(
    agent_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    current_user = _get_user_or_demo(authorize)
    if agent_id in current_user.owned_agents:
        raise HTTPException(
            status_code=400,
            detail="Owner cannot leave their own agent. Delete it instead.",
        )
    if agent_id not in current_user.collaborating_agents:
        raise HTTPException(status_code=404, detail="Collaboration not found")
    current_user.collaborating_agents.remove(agent_id)
    user_dao.set_user(current_user)
    return {"detail": "Left agent"}


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
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    try:
        _ensure_agent_owner(authorize, agent_id)
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
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    _ensure_agent_owner(authorize, agent_id)
    try:
        return access_service.revoke_key(agent_id, access_key_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/get-accesskeys", response_model=list[AccessKey])
def get_access_keys(
    agent_id: str,
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    _ensure_agent_owner(authorize, agent_id)
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
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    """Return all usable models for the requested provider using the supplied API key."""
    if os.getenv("DISABLE_AUTH", "").lower() != "true" and authorize is not None:
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
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
):
    """Return all usable embedding models for the requested provider using the supplied API key."""
    if os.getenv("DISABLE_AUTH", "").lower() != "true" and authorize is not None:
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
