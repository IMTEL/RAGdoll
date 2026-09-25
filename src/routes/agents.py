import logging
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel

from src.globals import access_service, agent_dao, user_dao
from src.llm import list_llm_models
from src.models.accesskey import AccessKey
from src.models.agent import Agent, Role
from src.models.errors.embedding_error import EmbeddingAPIError, EmbeddingError
from src.models.errors.llm_error import LLMAPIError
from src.models.model import Model
from src.models.users.user import User
from src.rag_service.embeddings import list_embedding_models
from src.utils.dependencies import has_agent_ownership, has_agent_access, current_user

router = APIRouter(tags=["Agents"])
logger = logging.getLogger(__name__)


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


class ExternalAgentInfo(BaseModel):
    agent_id: str
    name: str
    roles: list[Role]


def _access_key_valid_for_agent(agent: Agent, access_key: str | None) -> bool:
    normalized_key = access_key.strip() if access_key else None
    if not normalized_key:
        return False

    now = datetime.now()
    for stored_key in agent.access_key:
        if stored_key.key != normalized_key:
            continue
        return stored_key.expiry_date is None or now < stored_key.expiry_date
    return False


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


@router.post("/update-agent/", response_model=Agent)
def create_agent(
    agent: Agent, user: Annotated[User, Depends(current_user)]
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
            agent = agent_dao.add_agent(agent)
            # Add new agent id to owned agents
            user.owned_agents.append(agent.id)
            user_dao.set_user(user)
            return agent
        
        has_agent_access(agent.id, user)
        if not agent.llm_provider:
            agent.llm_provider = existing_agent.llm_provider
        if not agent.llm_model or agent.llm_model == "none":
            agent.llm_provider = existing_agent.llm_provider
            agent.llm_model = existing_agent.llm_model
        if not agent.llm_api_key:
            agent.llm_api_key = existing_agent.llm_api_key
        if not agent.embedding_api_key:
            agent.embedding_api_key = existing_agent.embedding_api_key
        agent.access_key = existing_agent.access_key

        updated_agent = agent_dao.add_agent(agent)
        if agent.id not in user.owned_agents:
            return _scrub_agent_api_keys(updated_agent)
        return updated_agent

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail="Invalid agentID, needs to be empty for new agents or an existing ID for updates",
        ) from e

@router.patch("/agents/{agent_id}", dependencies=[Depends(has_agent_access)], response_model=Agent)
@router.post("/agents/{agent_id}", dependencies=[Depends(has_agent_access)], response_model=Agent)
def update_agent(
    agent_id : str,
    agent: Agent, 
    user: Annotated[User, Depends(current_user)]
):
    """Update existing agent_id

    Args:
        agent (Agent): The agent configuration to create
        authorize (Annotated[AuthJWT, Depends()]): Jwt token object

    Returns:
        Agent: The created agent
    """
    try:
        # Check if agent exists
        existing_agent = agent_dao.get_agent_by_id(agent_id)
        if existing_agent is None:
            raise HTTPException(status_code=403)

        if not agent.llm_provider:
            agent.llm_provider = existing_agent.llm_provider
        if not agent.llm_model or agent.llm_model == "none":
            agent.llm_provider = existing_agent.llm_provider
            agent.llm_model = existing_agent.llm_model
        if not agent.llm_api_key:
            agent.llm_api_key = existing_agent.llm_api_key
        if not agent.embedding_api_key:
            agent.embedding_api_key = existing_agent.embedding_api_key
        agent.access_key = existing_agent.access_key

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
@router.get("/agents", response_model=list[Agent])
def get_agents(user: Annotated[User, Depends(current_user)]):
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """

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
@router.delete("/agents/{agent_id}", dependencies=[Depends(has_agent_ownership)])
def delete_agent(
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
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
    agent_dao.delete_agent_by_id(agent_id)
    if agent_id in user.owned_agents:
        user.owned_agents.remove(agent_id)
    user_dao.set_user(user)
    for collaborator in user_dao.get_users_with_agent(agent_id):
        if agent_id in collaborator.collaborating_agents:
            collaborator.collaborating_agents.remove(agent_id)
            user_dao.set_user(collaborator)


# Get a specific agent by ID
@router.get("/agents/{agent_id}", response_model=Agent, dependencies=[Depends(has_agent_access)])
def get_agent(
    agent_id : str,
    user: Annotated[User, Depends(current_user)],
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
    agent = agent_dao.get_agent_by_id(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    if agent_id not in user.owned_agents:
        return _scrub_agent_api_keys(agent)
    return agent


@router.get("/users/search", response_model=list[UserSearchResult], dependencies=[Depends(current_user)])
def search_users(
    q: str,
    limit: int = 10,
):
    users = user_dao.search_users(q, min(max(limit, 1), 25))
    return [
        UserSearchResult(**_public_user(user))
        for user in users
        if user.id is not None and user.id != user.id
    ]


@router.get(
    "/agents/{agent_id}/collaborators",
    response_model=CollaboratorsResponse,
    dependencies=[Depends(has_agent_access)]
)
def get_collaborators(
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
):

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
        current_user_id=user.id,
        is_owner=agent_id in user.owned_agents,
    )


@router.post(
    "/agents/{agent_id}/collaborators",
    response_model=CollaboratorsResponse,
    dependencies=[Depends(has_agent_ownership)]
)
def add_collaborator(
    agent_id: str,
    payload: CollaboratorInviteRequest,
    owner: Annotated[User, Depends(current_user)],
):
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
    return get_collaborators(agent_id, owner)

@router.delete(
    "/agents/{agent_id}/collaborators/{user_id}",
    response_model=CollaboratorsResponse,
    dependencies=[Depends(has_agent_ownership)]
)
def remove_collaborator(
    agent_id: str,
    user_id: str,
    owner: Annotated[User, Depends(current_user)],
):
    if owner and user_id == owner.id:
        raise HTTPException(status_code=400, detail="Owner cannot be removed")

    collaborator = user_dao.get_user_by_id(user_id)
    if collaborator is None:
        raise HTTPException(status_code=404, detail="User not found")
    if agent_id in collaborator.collaborating_agents:
        collaborator.collaborating_agents.remove(agent_id)
        user_dao.set_user(collaborator)
    return get_collaborators(agent_id, owner)


@router.post("/agents/{agent_id}/leave", dependencies=[Depends(has_agent_access)])
def leave_agent(
    agent_id: str,
    current_user: Annotated[User, Depends(current_user)],
):
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
    if access_key == None or not access_service.authenticate(agent_id, access_key):
        raise HTTPException(
            status_code=401, detail="Access key not valid for agent, Unauthorized"
        )

    agent = agent_dao.get_agent_by_id(agent_id)

    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id {agent_id} not found"
        )
    return agent


@router.get("/agent-info-by-accesskey", response_model=ExternalAgentInfo)
def agent_info_by_access_key(
    access_key: Annotated[str | None, Header()],
):
    """Resolve safe agent metadata from an external access key.

    This endpoint is intended for unauthenticated external clients. The access
    key itself is the authorization credential, so the response intentionally
    excludes model configuration, provider API keys, and access-key records.
    """
    agents = agent_dao.get_agents()
    total_keys = 0
    active_keys = 0
    expired_keys = 0
    has_header = bool(access_key and access_key.strip())
    now = datetime.now()

    for agent in agents:
        total_keys += len(agent.access_key)
        for stored_key in agent.access_key:
            if stored_key.expiry_date is None or now < stored_key.expiry_date:
                active_keys += 1
            else:
                expired_keys += 1
        if _access_key_valid_for_agent(agent, access_key):
            return ExternalAgentInfo(
                agent_id=agent.id or "",
                name=agent.name,
                roles=agent.roles,
            )

    logger.warning(
        "External access-key lookup failed. Header present: %s, agents scanned: %s, keys scanned: %s, active keys: %s, expired keys: %s",
        has_header,
        len(agents),
        total_keys,
        active_keys,
        expired_keys,
    )
    raise HTTPException(
        status_code=401,
        detail=(
            "Access key not valid. Ensure you are using the full key value shown "
            "when the key was created, not the key name or key id."
        ),
    )


# TODO : implement a better system of returning status codes on exceptions


@router.get("/new-accesskey", response_model=AccessKey)
def new_access_key(
    name: str,
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
    expiry_date: str | None = None,
    view_once: bool = True,
):
    has_agent_ownership(agent_id, user)
    try:
        if expiry_date is None:
            return access_service.generate_accesskey(name, None, agent_id, view_once)
        else:
            expiry_date_formatted = datetime.fromisoformat(expiry_date).replace(
                tzinfo=None
            )
            return access_service.generate_accesskey(
                name, expiry_date_formatted, agent_id, view_once
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/revoke-accesskey")
def revoke_access_key(
    access_key_id: str,
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
):
    has_agent_access(agent_id, user);
    try:
        return access_service.revoke_key(agent_id, access_key_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get("/get-accesskeys", response_model=list[AccessKey])
def get_access_keys(
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
):
    has_agent_access(agent_id, user)
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f" agent of id not found {agent_id}"
        )
    access_keys = [access_key.model_copy() for access_key in agent.access_key]
    for access_key in access_keys:
        if access_key.view_once:
            access_key.key = None
    return access_keys


@router.get("/chat-accesskey", response_model=AccessKey)
@router.get("/chat-access-key", response_model=AccessKey)
def chat_access_key(
    agent_id: str,
    user: Annotated[User, Depends(current_user)],
):
    has_agent_access(agent_id, user);
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f" agent of id not found {agent_id}"
        )

    user_key_suffix = user.id if user and user.id else "demo"
    key_name = f"Chat Access Key - {user_key_suffix}"
    now = datetime.now()

    for access_key in agent.access_key:
        if (
            access_key.name == key_name
            and access_key.key
            and (access_key.expiry_date is None or access_key.expiry_date > now)
        ):
            return access_key

    return access_service.generate_accesskey(
        key_name,
        now + timedelta(days=2),
        agent_id,
    )


@router.post("/get_models", response_model=list[Model])
def fetch_models(
    payload: ProviderKeyRequest,
    user: Annotated[AuthJWT | None, Depends(current_user)] = None,
):
    try:
        return list_llm_models(payload.provider, payload.api_key)
    except LLMAPIError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/get_embedding_models", response_model=list[str], dependencies=[Depends(current_user)])
def fetch_embedding_models(
    payload: ProviderKeyRequest,
):
    try:
        return list_embedding_models(payload.provider, payload.api_key)
    except EmbeddingAPIError as error:
        status_code = _map_embedding_api_error(error)
        raise HTTPException(status_code=status_code, detail=str(error)) from error
    except EmbeddingError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
