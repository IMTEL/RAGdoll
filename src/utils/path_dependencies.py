import logging
import re
from typing import Annotated, Literal

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import Config
from src.globals import access_service, agent_dao, auth_service
from src.models.agent import Agent
from src.models.users.user import User


logger = logging.getLogger(__name__)

_agent_not_found = HTTPException(status_code=404, detail="Agent not found.")
_not_authorized = HTTPException(status_code=403, detail="Unauthorized")
_not_authenticated = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)
_security = HTTPBearer()
# Three non-empty base64url segments separated by dots: header.payload.signature
_JWT_SHAPE = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")


def get_auth(
    auth_header: Annotated[HTTPAuthorizationCredentials, Depends(_security)],
) -> str:
    """Gets the authentication headers. Returns either JWT token if it is one or a access token. Both are string.

    Args:
        request (Request): The http request
    Raises:
        HTTPException: 403 if the authorization is not there
        HTTPException: 400 if the authorization is invalid with the "Bearer " prefix
    Returns:
        str : the string that is after the bearer prefix
    """
    if not auth_header.credentials:
        # If no header and auth is required, AuthJWT will handle the error
        raise _not_authorized
    return auth_header.credentials


def get_bearer_token(request: Request) -> str:
    """Extracts the bearer credentials from a request's Authorization header.

    For call sites that cannot declare ``get_auth`` as a dependency.

    Args:
        request (Request): The http request

    Raises:
        HTTPException: 401 if the header is missing, not a Bearer scheme, or empty

    Returns:
        str: the string that is after the bearer prefix
    """
    scheme, _, credentials = request.headers.get("Authorization", "").partition(" ")
    credentials = credentials.strip()
    if scheme.lower() != "bearer" or not credentials:
        raise _not_authenticated
    return credentials


def is_jwt_shaped(token: str) -> bool:
    """Returns True if the token has the header.payload.signature JWT shape."""
    return _JWT_SHAPE.fullmatch(token) is not None


def get_optional_user(
    authorize: Annotated[str, Depends(get_auth)],
) -> User | str:
    """Resolves the bearer string to a user, or returns it as an access key.

    JWT-shaped strings are always validated as user tokens; only strings that are
    not JWT-shaped are treated as access keys.

    Args:
        authorize (Annotated[str, Depends): the string from the authorization headers

    Raises:
        HTTPException: 401 if a JWT-shaped token fails validation (expired, bad
            signature, identity provider unreachable)

    Returns:
        User | str: the authenticated user, or the string as an access key
    """
    if not is_jwt_shaped(authorize):
        return authorize
    try:
        return auth_service.get_authenticated_user(authorize)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Rejected bearer token that failed JWT validation: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


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


def agent_with_id(agent_id: str) -> Agent:
    """Can be used as a dependency with the argument "agent_id".

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
) -> Agent:
    """Checks if an access key or a user has call access to the agent.

    Args:
        agent_id (str): the agent id
        authorize (Annotated[User | str, Depends): the user, or the access key

    Raises:
        HTTPException: 403 if the access key is not valid for the agent
        HTTPException: 404 if the user has no access to the agent, or it does not exist

    Returns:
        Agent: The agent they have access to
    """
    if isinstance(authorize, User):
        require_user_has_agent_access(agent_id, authorize)
    elif not access_service.authenticate(agent_id, authorize):
        raise _not_authorized
    return agent_with_id(agent_id)


def require_user_or_access_key(
    request: Request,
    x_agent_id: Annotated[str | None, Header()] = None,
) -> User | Agent:
    """Requires a user token, or an access key for the agent in ``X-Agent-Id``.

    The credentials come from the ``Authorization: Bearer`` header. A JWT-shaped
    token must be a valid user token. Anything else is treated as an access key
    and checked against the agent named in the ``X-Agent-Id`` header.

    Args:
        request (Request): The http request
        x_agent_id (str | None): the agent id from the ``X-Agent-Id`` header

    Raises:
        HTTPException: 401 if the credentials are missing, the user token is
            invalid, or an access key is sent without an ``X-Agent-Id`` header
        HTTPException: 403 if the access key is not valid for the agent

    Returns:
        User | Agent: the authenticated user, or the agent the key unlocks
    """
    authorize = get_optional_user(get_bearer_token(request))
    if isinstance(authorize, User):
        return authorize
    if not x_agent_id:
        raise HTTPException(
            status_code=401,
            detail="An access key must be sent with an X-Agent-Id header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return has_agent_call_access(x_agent_id, authorize)


def require_speech_access(
    request: Request,
    x_agent_id: Annotated[str | None, Header()] = None,
) -> User | Agent | None:
    """``require_user_or_access_key`` unless PUBLIC_SPEECH_ENDPOINTS is set.

    Only for the speech and translation routes, which XR/Unity clients may call
    without credentials when the deployment opts in.

    Returns:
        User | Agent | None: None when the routes are public
    """
    if Config().PUBLIC_SPEECH_ENDPOINTS:
        return None
    return require_user_or_access_key(request, x_agent_id)
