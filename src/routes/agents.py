import datetime
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.access_service.access_service import (
    AccessKeyNotFoundError,
    AccessKeyNotRevealableError,
    AccessKeyUndecryptableError,
    parse_access_key,
)
from src.globals import access_service, agent_dao, user_dao
from src.models.accesskey import AccessKeyCreated, AccessKeyPublic, as_utc
from src.models.agent import Agent, AgentPublic, Role
from src.models.users.user import User
from src.utils.path_dependencies import (
    agent_with_id,
    get_auth,
    get_optional_user,
    has_agent_call_access,
    require_user,
    require_user_has_agent_access,
    require_user_has_agent_ownership,
)

# Bad practice, but makes every dependency accessible without 6 different imports
from src.utils.provider_keys import (
    ALLOWED_USAGES,
    PROVIDER_KEY_FIELDS,
    resolve_provider_keys,
)


router = APIRouter(prefix="/agents", tags=["Agents"])
logger = logging.getLogger(__name__)

CHAT_ACCESS_KEY_LIFETIME = datetime.timedelta(hours=24)


def _public_user(user: User, role: str | None = None) -> dict:
    return {
        "id": user.id or "",
        "name": user.name,
        "email": user.email,
        "picture": user.picture,
        "role": role,
    }


def _agent_for_access_key_holder(agent: Agent) -> AgentPublic:
    """Agent view for access-key callers: no provider keys and no access keys."""
    return AgentPublic.from_agent(agent, include_access_keys=False)


def _public(agent: Agent) -> AgentPublic:
    """Agent view for users, without secrets.

    Stored-key references are resolved only to set the ``*_invalid`` flags.
    """
    return AgentPublic.from_agent(resolve_provider_keys(agent, user_dao))


def _apply_provider_key_choice(
    new: Agent, existing: Agent | None, user: User, is_owner: bool
) -> None:
    """Applies the provider-key rules of a create or update to ``new``, in place.

    Per key (LLM and embedding):

    - A non-empty literal key wins and clears the stored-key reference.
    - Otherwise a reference (``*_api_key_id``) to a stored key replaces the
      literal key. Stored keys belong to users, and references always resolve
      against the agent owner's keys, so only the owner may set or change a
      reference, and only to one of their own keys of a matching usage.
      Re-sending the agent's current reference is allowed for collaborators.
    - Neither: keep what is stored (the API never returns keys to send back).

    Raises:
        HTTPException: 403 if a collaborator changes the reference, 404 if the
            key is not one of the owner's stored keys, 400 if its usage does
            not fit (an embedding-only key as the LLM key, or vice versa)
    """
    for field in PROVIDER_KEY_FIELDS:
        id_field = f"{field}_id"
        key_id = getattr(new, id_field)
        if getattr(new, field):
            setattr(new, id_field, None)
            continue
        current_id = getattr(existing, id_field) if existing else None
        if not key_id:
            if existing is not None:
                setattr(new, field, getattr(existing, field))
                setattr(new, id_field, current_id)
            continue
        if key_id != current_id:
            if not is_owner:
                raise HTTPException(
                    status_code=403,
                    detail="Only the agent owner can select a stored API key",
                )
            stored = next((k for k in user.api_keys if k.id == key_id), None)
            if stored is None:
                raise HTTPException(status_code=404, detail="API key not found")
            if stored.usage not in ALLOWED_USAGES[field]:
                raise HTTPException(
                    status_code=400,
                    detail=f"API key '{stored.label}' cannot be used as {field}",
                )
        setattr(new, field, "")


def _may_hold_access_key(agent: Agent, access_key: str) -> bool:
    """Cheap pre-filter before ``authenticate`` when scanning every agent.

    New-format keys name their key id, so only agents with that id can match.
    Legacy keys carry no id and must be checked against every agent.
    """
    key_id = parse_access_key(access_key)
    return key_id is None or any(ak.id == key_id for ak in agent.access_key)


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


@router.post("", response_model=AgentPublic)
def create_agent(
    agent: Agent, user: Annotated[User, Depends(require_user)]
) -> AgentPublic:
    """Creates a new agent.

    Args:
        agent (Agent): The agent that will be created. ID will be generated when creating
        user (Annotated[User, Depends): The user fetched by the authorization token

    Raises:
        HTTPException: 400/404 for an unusable ``*_api_key_id`` (see
            ``_apply_provider_key_choice``)

    Returns:
        AgentPublic: the created agent, without secrets
    """
    agent.id = None
    # Access keys are only created through the access-key endpoints.
    agent.access_key = []
    _apply_provider_key_choice(agent, None, user, is_owner=True)
    agent = agent_dao.add_agent(agent)
    # Add new agent id to owned agents
    user.owned_agents.append(agent.id)
    user_dao.set_user(user)
    return _public(agent)


@router.patch(
    "/{agent_id}",
    response_model=AgentPublic,
    dependencies=[Depends(require_user_has_agent_access)],
)
@router.post(
    "/{agent_id}",
    response_model=AgentPublic,
    dependencies=[Depends(require_user_has_agent_access)],
)
def update_agent(
    updated_agent: Agent,
    existing_agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(require_user)],
) -> AgentPublic:
    """Update existing agent.

    An empty ``llm_api_key`` or ``embedding_api_key`` keeps the stored key,
    since the API never returns provider keys to send back. See
    ``_apply_provider_key_choice`` for stored-key references.

    Args:
        updated_agent (Agent): The updated agent
        existing_agent (Agent): The existing agent within the repository
        user (User): The current user

    Raises:
        HTTPException: 404 if the AgentID is invalid
        HTTPException: 400/403/404 for an unusable ``*_api_key_id``

    Returns:
        AgentPublic: the updated agent, without secrets
    """
    updated_agent.id = existing_agent.id
    if not updated_agent.llm_provider:
        updated_agent.llm_provider = existing_agent.llm_provider
    if not updated_agent.llm_model or existing_agent.llm_model == "none":
        updated_agent.llm_provider = existing_agent.llm_provider
        updated_agent.llm_model = existing_agent.llm_model
    _apply_provider_key_choice(
        updated_agent,
        existing_agent,
        user,
        is_owner=existing_agent.id in user.owned_agents,
    )
    updated_agent.access_key = existing_agent.access_key

    updated_agent = agent_dao.add_agent(updated_agent)
    return _public(updated_agent)


# Get all agents
@router.get("", response_model=list[AgentPublic] | list[ExternalAgentInfo])
def get_agents(access: Annotated[User | str, Depends(get_optional_user)]):
    """Retrieve the agents the caller can see.

    Returns:
        list[AgentPublic]: the user's owned and shared agents, without secrets
        list[ExternalAgentInfo]: for an access key, the agent it unlocks
    """
    agents: list = []
    if isinstance(access, User):
        agents += [
            _public(agent)
            for agent_id in [*access.owned_agents, *access.collaborating_agents]
            if (agent := agent_dao.get_agent_by_id(agent_id)) is not None
        ]
    elif isinstance(access, str):
        all_agents = agent_dao.get_agents()
        for agent in all_agents:
            if _may_hold_access_key(agent, access) and access_service.authenticate(
                agent.id, access
            ):
                return [
                    ExternalAgentInfo(
                        agent_id=agent.id or "", name=agent.name, roles=agent.roles
                    )
                ]
    return agents


# Get a specific agent by ID
@router.delete("/{agent_id}", dependencies=[Depends(require_user_has_agent_ownership)])
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


@router.get(
    "/users/search",
    response_model=list[UserSearchResult],
)
def search_users(
    q: str,
    requester: Annotated[User, Depends(require_user)],
    limit: int = 10,
):
    """Searches users to invite as collaborators, excluding the requesting user."""
    users = user_dao.search_users(q, min(max(limit, 1), 25))
    return [
        UserSearchResult(**_public_user(user))
        for user in users
        if user.id is not None and user.id != requester.id
    ]


@router.get(
    "/{agent_id}/collaborators",
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
    "/{agent_id}/collaborators",
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
    "/{agent_id}/collaborators/{user_id}",
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


@router.post("/{agent_id}/leave", dependencies=[Depends(require_user_has_agent_access)])
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
    "/agent-info",
    response_model=AgentPublic,
    dependencies=[Depends(has_agent_call_access)],
)
def agent_info(
    agent_id: str,
    access_key: Annotated[str, Depends(get_auth)],
):
    """Retrieve a specific agent by ID.

    Args:
        agent_id (str): The unique identifier of the agent
        access_key (Annotated[str | None, Header()]): access key header

    Returns:
        AgentPublic: The requested agent, without provider API keys or access keys

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
    return _agent_for_access_key_holder(agent)


@router.get("/agent-info-by-accesskey", response_model=ExternalAgentInfo)
def agent_info_by_access_key(
    access_key: Annotated[str, Depends(get_auth)],
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
    now = datetime.datetime.now(datetime.UTC)
    if not (access_key and access_key.strip()):
        raise HTTPException(
            status_code=400,
            detail="You need a access token in the authorization header to fetch agents",
        )
    for agent in agents:
        total_keys += len(agent.access_key)
        for stored_key in agent.access_key:
            expiry_date = as_utc(stored_key.expiry_date)
            if expiry_date is None or now < expiry_date:
                active_keys += 1
            else:
                expired_keys += 1
        if (
            agent.id
            and _may_hold_access_key(agent, access_key)
            and access_service.authenticate(agent.id, access_key)
        ):
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


@router.get("/new-accesskey", response_model=AccessKeyCreated)
@router.post(
    "/{agent_id}/accesskeys/",
    response_model=AccessKeyCreated,
    dependencies=[Depends(require_user_has_agent_ownership)],
)
def new_access_key(
    name: str,
    agent_id: str,
    user: Annotated[User, Depends(require_user)],
    expiry_date: str | None = None,
    view_once: bool = True,
) -> AccessKeyCreated:
    """Creates an access key and returns it.

    ``expiry_date`` is ISO 8601; without a timezone it is taken as UTC. With
    ``view_once`` (the default) this response is the only time the key is
    shown. With ``view_once=false`` the agent's owner and collaborators can
    show it again via ``GET /agents/{agent_id}/accesskeys/{key_id}/reveal``.
    """
    require_user_has_agent_ownership(agent_id, user)
    try:
        expiry = (
            None
            if expiry_date is None
            else datetime.datetime.fromisoformat(expiry_date)
        )
        return access_service.generate_accesskey(
            name, expiry, agent_id, view_once=view_once
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}") from e


@router.delete(
    "/{agent_id}/accesskeys/{access_key_id}",
    response_model=bool,
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


class RevealedAccessKey(BaseModel):
    key: str


@router.get(
    "/{agent_id}/accesskeys/{access_key_id}/reveal",
    response_model=RevealedAccessKey,
    dependencies=[Depends(require_user_has_agent_access)],
)
def reveal_access_key(
    access_key_id: str,
    agent: Annotated[Agent, Depends(agent_with_id)],
) -> RevealedAccessKey:
    """Shows a key created with ``view_once=false`` again.

    Only for logged-in owners and collaborators of the agent: callers using an
    access key get 403 from ``require_user``.
    """
    try:
        return RevealedAccessKey(key=access_service.reveal_key(agent.id, access_key_id))
    except AccessKeyNotFoundError:
        raise HTTPException(status_code=404, detail="Access key not found") from None
    except AccessKeyUndecryptableError:
        raise HTTPException(
            status_code=409,
            detail="This key can no longer be revealed; create a new one.",
        ) from None
    except AccessKeyNotRevealableError:
        raise HTTPException(
            status_code=409,
            detail=(
                "This key was created to be shown only once and cannot be "
                "revealed. Create a new key if you lost it."
            ),
        ) from None


@router.get(
    "/{agent_id}/accesskeys",
    response_model=list[AccessKeyPublic],
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
    return [AccessKeyPublic.model_validate(ak.model_dump()) for ak in agent.access_key]


@router.get("/chat-accesskey", response_model=AccessKeyCreated)
@router.get("/chat-access-key", response_model=AccessKeyCreated)
@router.post(
    "/{agent_id}/accesskeys",
    response_model=AccessKeyCreated,
    dependencies=[Depends(require_user_has_agent_access)],
)
def chat_access_key(
    agent: Annotated[Agent, Depends(agent_with_id)],
    user: Annotated[User, Depends(require_user)],
) -> AccessKeyCreated:
    """Mints a new 24-hour access key for the user to chat with the agent.

    Stored keys cannot be shown again, so every call creates a new key. The
    user's expired chat keys are removed first so they do not pile up.
    """
    require_user_has_agent_access(
        agent.id, user
    )  # Still in place because of the get routes

    user_key_suffix = user.id if user and user.id else "demo"
    key_name = f"Chat Access Key - {user_key_suffix}"
    now = datetime.datetime.now(datetime.UTC)

    expired_ids = [
        access_key.id
        for access_key in agent.access_key
        if access_key.name == key_name
        and as_utc(access_key.expiry_date) is not None
        and as_utc(access_key.expiry_date) <= now
    ]
    if expired_ids:
        # One $pull: keys minted or edits made since this read are kept.
        access_service.revoke_keys(agent.id, expired_ids)

    return access_service.generate_accesskey(
        key_name, now + CHAT_ACCESS_KEY_LIFETIME, agent.id
    )


# Get a specific agent by ID.
# Registered last: GET "/{agent_id}" would otherwise swallow the literal GET
# paths above (e.g. "/agent-info-by-accesskey", "/chat-accesskey").
@router.get(
    "/{agent_id}",
    response_model=AgentPublic,
    dependencies=[Depends(has_agent_call_access)],
)
def get_agent(
    agent: Annotated[Agent, Depends(agent_with_id)],
    caller: Annotated[User | str, Depends(get_optional_user)],
):
    """Retrieve a specific agent by ID.

    Args:
        agent (Agent): The agent, resolved from the agent_id path parameter
        caller (Annotated[User | str, Depends()]): The user or access key from the token

    Returns:
        AgentPublic: The requested agent, without secrets. Access-key callers
            do not see the agent's access keys at all.
    """
    if not isinstance(caller, User):
        return _agent_for_access_key_holder(agent)
    return _public(agent)
