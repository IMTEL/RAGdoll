import datetime
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from src.globals import access_service, agent_dao, user_dao
from src.models.accesskey import AccessKey
from src.models.agent import Agent, Role
from src.models.users.user import User

# Bad practice, but makes every dependency accessible without 6 different imports
from src.utils.path_dependencies import (
    agent_with_id,
    get_optional_user,
    has_agent_call_access,
    require_user,
    require_user_has_agent_access,
    require_user_has_agent_ownership,
)


router = APIRouter(prefix="/agents", tags=["Agents"])
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
    get_user_id: str | None = None
    is_owner: bool = False


class ProviderKeyRequest(BaseModel):
    provider: str
    api_key: str


class ExternalAgentInfo(BaseModel):
    agent_id: str
    name: str
    roles: list[Role]


@router.post("/agents")
def create_agent(agent: Agent, user: Annotated[User, Depends(require_user)]) -> Agent:
    """Creates a new agent.

    Args:
        agent (Agent): The agent that will be created. ID will be generated when creating
        user (Annotated[User, Depends): The user fetched by the authorization token

    Raises:
        HTTPException: If

    Returns:
        Agent: _description_
    """
    agent.id = None
    agent = agent_dao.add_agent(agent)
    # Add new agent id to owned agents
    user.owned_agents.append(agent.id)
    user_dao.set_user(user)
    return agent


@router.patch(
    "/agents/{agent_id}",
    dependencies=[Depends(require_user_has_agent_access)],
)
@router.post(
    "/agents/{agent_id}",
    dependencies=[Depends(require_user_has_agent_access)],
)
def update_agent(
    updated_agent: Agent,
    existing_agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(require_user)],
) -> Agent:
    """Update existing agent.

    Args:
        updated_agent (Agent): The updated agent
        existing_agent (Agent): The existing agent within the repository
        user (Annotated[User, Depends): The current user

    Raises:
        HTTPException: 404 if the AgentID is invalid
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    updated_agent.id = existing_agent.id
    if not updated_agent.llm_provider:
        updated_agent.llm_provider = existing_agent.llm_provider
    if not updated_agent.llm_model or existing_agent.llm_model == "none":
        updated_agent.llm_provider = existing_agent.llm_provider
        updated_agent.llm_model = existing_agent.llm_model
    if not updated_agent.llm_api_key:
        updated_agent.llm_api_key = existing_agent.llm_api_key
    if not updated_agent.embedding_api_key:
        updated_agent.embedding_api_key = existing_agent.embedding_api_key
    updated_agent.access_key = existing_agent.access_key

    updated_agent = agent_dao.add_agent(updated_agent)
    if updated_agent.id not in user.owned_agents:
        return _scrub_agent_api_keys(updated_agent)
    return updated_agent


# Get all agents
@router.get("/agents", response_model=list[Agent])
def get_agents(access: Annotated[User | str, Depends(get_optional_user)]):
    """Retrieve all agent configurations.

    Returns:
        list[Agent]: All stored agents
    """
    agents: list = []
    if isinstance(access, User):
        agents += [
            agent
            for agent_id in access.owned_agents
            if (agent := agent_dao.get_agent_by_id(agent_id)) is not None
        ]
        agents += [
            _scrub_agent_api_keys(agent)
            for agent_id in access.collaborating_agents
            if (agent := agent_dao.get_agent_by_id(agent_id)) is not None
        ]
    elif isinstance(access, str):
        all_agents = agent_dao.get_agents()
        for agent in all_agents:
            if access_service.authenticate(agent.id, access):
                return [
                    ExternalAgentInfo(
                        agent_id=agent.id or "", name=agent.name, roles=agent.roles
                    )
                ]
    return agents


# Get a specific agent by ID
@router.delete(
    "/agents/{agent_id}", dependencies=[Depends(require_user_has_agent_ownership)]
)
def delete_agent(
    agent_id: str,
    user: Annotated[User, Depends(require_user)],
):
    """Deletes a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        user (Annotated[User, Depends(require_user)]): The user from the authorization token
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
@router.get(
    "/agents/{agent_id}",
    response_model=Agent,
    dependencies=[Depends(has_agent_call_access)],
)
def get_agent(
    agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(get_optional_user)],
):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        authorize (Annotated[User, Depends()]): The user fetched from the token

    Returns:
        Agent: The requested agent
    """
    if agent.id not in user.owned_agents:
        return _scrub_agent_api_keys(agent)
    return agent


@router.get(
    "/users/search",
    response_model=list[UserSearchResult],
    dependencies=[Depends(require_user)],
)
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
    dependencies=[Depends(require_user_has_agent_access)],
)
def get_collaborators(
    agent_id: str,
    user: Annotated[User, Depends(require_user)],
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
        get_user_id=user.id,
        is_owner=agent_id in user.owned_agents,
    )


@router.post(
    "/agents/{agent_id}/collaborators",
    response_model=CollaboratorsResponse,
    dependencies=[Depends(require_user_has_agent_ownership)],
)
def add_collaborator(
    agent_id: str,
    payload: CollaboratorInviteRequest,
    owner: Annotated[User, Depends(require_user)],
):
    """Adds a collaborator to the agent.

    Args:
        agent_id (str): the agent_id
        payload (CollaboratorInviteRequest): the
        owner (Annotated[User, Depends): _description_

    Raises:
        HTTPException: 404 if the user is not found
        HTTPException: 400 if the user is the owner
        HTTPException: 400 if the user already owns the agent

    Returns:
        _type_: _description_
    """
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
    dependencies=[Depends(require_user_has_agent_ownership)],
)
def remove_collaborator(
    agent_id: str,
    user_id: str,
    owner: Annotated[User, Depends(require_user)],
):

    collaborator = user_dao.get_user_by_id(user_id)
    if collaborator is None:
        raise HTTPException(status_code=404, detail="User not found")
    if agent_id in collaborator.collaborating_agents:
        collaborator.collaborating_agents.remove(agent_id)
        user_dao.set_user(collaborator)
    return get_collaborators(agent_id, owner)


@router.post(
    "/agents/{agent_id}/leave", dependencies=[Depends(require_user_has_agent_access)]
)
def leave_agent(
    agent_id: str,
    user: Annotated[User, Depends(require_user)],
):
    """Leaves the user from the agent.

    Args:
        agent_id (str): the string of the agent id
        user (Annotated[User, Depends): The user fetched from the Authorization headers

    Raises:
        HTTPException: 400 if the user is the owner
        HTTPException: 404 if the user is not a collaborator of the agent

    Returns:
        _type_: _description_
    """
    if agent_id in user.owned_agents:
        raise HTTPException(
            status_code=400,
            detail="Owner cannot leave their own agent. Delete it instead.",
        )
    if agent_id not in user.collaborating_agents:
        raise HTTPException(status_code=404, detail="Collaboration not found")
    user.collaborating_agents.remove(agent_id)
    user_dao.set_user(user)
    return {"detail": "Left agent"}


# Get a specific agent by ID using AccessKey for authentication
@router.get(
    "/agent-info", response_model=Agent, dependencies=[Depends(has_agent_call_access)]
)
@router.get("/agents/{agent_id}", response_model=Agent)
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
    now = datetime.datetime.now(tz=datetime.UTC)
    if not (access_key and access_key.strip()):
        raise HTTPException(
            status_code=400,
            detail="You need a access token in the authorization header to fetch agents",
        )
    for agent in agents:
        total_keys += len(agent.access_key)
        for stored_key in agent.access_key:
            if stored_key.expiry_date is None or now < stored_key.expiry_date:
                active_keys += 1
            else:
                expired_keys += 1
        if access_service.authenticate(agent, access_key):
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
    user: Annotated[User, Depends(require_user)],
    expiry_date: str | None = None,
    view_once: bool = True,
):
    require_user_has_agent_ownership(agent_id, user)
    try:
        if expiry_date is None:
            return access_service.generate_accesskey(name, None, agent_id, view_once)
        else:
            expiry_date_formatted = datetime.datetime.fromisoformat(
                expiry_date
            ).replace(tzinfo=None)
            return access_service.generate_accesskey(
                name, expiry_date_formatted, agent_id, view_once
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.delete(
    "/agents/{agent_id}/accesskeys/{access_key_id}",
    response_model=list[AccessKey],
    dependencies=[Depends(require_user_has_agent_access)],
)
def revoke_access_key(
    access_key_id: str,
    agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(require_user)],
):
    # Still here because old path
    require_user_has_agent_access(agent.id, user)
    try:
        return access_service.revoke_key(agent.id, access_key_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.get(
    "/agents/{agent_id}/accesskeys",
    response_model=list[AccessKey],
    dependencies=[Depends(require_user_has_agent_access)],
)
def get_access_keys(
    agent_id: str,
    user: Annotated[User, Depends(require_user)],
):
    require_user_has_agent_access(agent_id, user)
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
@router.post(
    "/agents/{agent_id}/accesskeys",
    response_model=AccessKey,
    dependencies=[Depends(require_user_has_agent_access)],
)
def chat_access_key(
    agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(require_user)],
):
    require_user_has_agent_access(
        agent.id, user
    )  # Still in place because of the get routes

    user_key_suffix = user.id if user and user.id else "demo"
    key_name = f"Chat Access Key - {user_key_suffix}"
    now = datetime.datetime.now(tz=datetime.UTC)

    for access_key in agent.access_key:
        if (
            access_key.name == key_name
            and access_key.key
            and (access_key.expiry_date is None or access_key.expiry_date > now)
        ):
            return access_key

    return access_service.generate_accesskey(
        key_name,
        now + datetime.timedelta(days=2),
        agent.id,
    )
