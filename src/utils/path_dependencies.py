import contextlib
import datetime
from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request

from src.globals import agent_dao, auth_service
from src.models.agent import Agent
from src.models.users.user import User


_agent_not_found = HTTPException(status_code=404, detail="Agent not found.")
_not_authorized = HTTPException(status_code=403, detail="Unauthorized")


def get_auth(request: Request) -> str:
    """Gets the authentication headers. Returns either JWT token if it is one or a access token. Both are string.

    Args:
        request (Request): The http request
    Raises:
        HTTPException: 403 if the authorization is not there
        HTTPException: 400 if the authorization is invalid with the "Bearer " prefix
    Returns:
        str : the string that is after the bearer prefix
    """
    auth_header = request.headers.get("authorization")
    if not auth_header:
        # If no header and auth is required, AuthJWT will handle the error
        raise _not_authorized
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=400, detail="Invalid header bearer token")
    return auth_header[7:]


def get_optional_user(
    authorize: Annotated[str, Depends(get_auth)],
) -> User | str:
    """Tries to fetch the User from the string. Returns either string which should be thought as a access token or a user.

    Args:
        authorize (Annotated[str, Depends): the string from the authorization headers

    Returns:
        User | str: A user if found which is valid or the string that was tried to fetch
    """
    user = authorize
    with contextlib.suppress(Exception):
        user = auth_service.get_authenticated_user(authorize)
    return user


def require_user(maybe_user: Annotated[User | str, Depends(get_optional_user)]) -> User:
    """Enforces the authorization header to be a user.

    Args:
        maybe_user (Annotated[User  |  str, Depends): The authorization header

    Raises:
        HTTPException: 403, if the token is not a active user token

    Returns:
        User: The typecasting to user
    """
    if not isinstance(maybe_user, User):
        raise HTTPException(
            status_code=403, detail="You need to be a user to access this path"
        )
    return maybe_user


def require_user_has_agent_ownership(
    agent_id: str, user: Annotated[User, Depends(require_user)]
) -> Literal[True]:
    """Ensures the user has ownership of a specific agent_id.

    Args:
        agent_id (str): the agent_id
        user (Annotated[User, Depends): the user fetched from the authorization header

    Raises:
        HTTPException: 404 if the agent can not be found in the owner

    Returns:
        Literal[True]: _description_
    """
    if agent_id not in user.owned_agents:
        raise _agent_not_found
    return True


def require_user_has_agent_access(
    agent_id: str, user: Annotated[User, Depends(require_user)]
) -> Literal[True]:
    """Checks if a user has access.

    Throws 404 if the user does not have access and returns True else.

    Args:
        agent_id (str): the id of the agent
        user (User): The user that wants to access the
    Raises:
        HTTPException: 404 if the agent can not be found or the
    Returns:
        Literal[True]: Returns true if the
    """
    if not (
        agent_id in user.collaborating_agents
        or require_user_has_agent_ownership(agent_id, user)
    ):
        raise _agent_not_found
    return True


def _contains_valid_access_key(agent: Agent, access_key: str) -> bool:
    """Helper function to check if a access key is contained within a agent.

    Args:
        agent (Agent): The agent
        access_key (str): The access_key

    Returns:
        bool: returns true or false depending if the access key is valid for the agent
    """
    for key in agent.access_key:
        if key.key == access_key:
            return key.expiry_date == None or (
                key.expiry_date > datetime.datetime.now(tz=datetime.UTC)
            )
    return False


def agent_with_id(agent_id: str) -> Agent:
    """Can be used as a dependency with the arguemnt "agent_id".

    Args:
        agent_id (str): A agent ID

    Raises:
        HTTPException: 404 if agent can not be found.

    Returns:
        Agent: the agent with the specific id.
    """
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent == None:
        raise _agent_not_found
    return agent


def has_agent_call_access(
    agent_id: str, authorize: Annotated[User | str, Depends(get_optional_user)]
) -> Literal[True]:
    """Checks if a access token or a user token has access token to the agent.

    Args:
        agent_id (str): the agent id
        authorize (Annotated[str, Depends): the token or user

    Raises:
        HTTPException: 403 if the user or access token is not authorized.

    Returns:
        Literal[True]: _description_
    """
    if authorize == None:
        raise _not_authorized
    if isinstance(authorize, User):
        require_user_has_agent_access(agent_id, authorize)
    else:
        agent = agent_with_id(agent_id)
        if not _contains_valid_access_key(agent, authorize):
            raise _not_authorized
    return True
