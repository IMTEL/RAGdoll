from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request
from fastapi_jwt_auth import AuthJWT

from src.globals import auth_service, agent_dao
from src.models.agent import Agent
from src.models.users.user import User

def get_auth(request: Request) -> AuthJWT | str | None:
    """
    Gets the authentication headers. 
    Returns either JWT token if it is one or a access token 

    Args:
        request (Request): The http request

    Raises:
        HTTPException: 403

    Returns:
        AuthJWT | str : 
    """
    key  = request.headers.get("authorization")
    if not key:
        raise HTTPException(status_code=403, detail="Unauthorized access")
    try:
        return AuthJWT(request)
    except Exception:
        return key

def get_optional_user(authorize: Annotated[AuthJWT | str, Depends(get_auth)]) -> User | str:
    """
    If you want to ensure only a user can be fetched use get_user instead
    
    Dependency allowing for fetching of the current user
    Return the User or the access token
    """
    if not isinstance(authorize, AuthJWT):
        return authorize
    return auth_service.get_authenticated_user(authorize);

def require_user(maybeUser : Annotated[User | str, Depends(get_optional_user)]) -> User:
    """
        Helper function to enforce a user type if there is a user
    """
    if not isinstance(maybeUser, User):
        raise HTTPException(status_code=403, detail="You need to be a user to access this path")
    return maybeUser


def require_user_has_agent_ownership(agent_id : str, user : Annotated[User, Depends(require_user)]) -> Literal[True]:
    """
        Checks if the user owns the agent
        Throws 404 if the user does not own it and returns True else
    """
    if not agent_id in user.owned_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return True

def require_user_has_agent_access(agent_id : str, user : Annotated[User, Depends(require_user)]) -> Literal[True]:
    """
        Checks if a user has access
        Throws 404 if the user does not have access and returns True else
    """
    if not (agent_id in user.collaborating_agents or require_user_has_agent_ownership(agent_id, user)):
        raise HTTPException(status_code=404, detail="Agent not found") 
    return True

def _contains_access_key(agent : Agent, access_key : str) -> bool:
    """
    Helper function to check if a access key is contained within a agent
    Args:
        agent (Agent): _description_
        access_key (str): _description_

    Returns:
        bool: _description_
    """
    for key in agent.access_key:
        if key.key == access_key:
            return True
    return False


def agent_with_id(agent_id : str) -> Agent:
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent == None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


def has_agent_call_access(agent_id: str, authorize : Annotated[str | AuthJWT, Depends(get_auth)]) -> Literal[True]:
    """
        Checks if a user or access key can fetch the information from the agent
    """
    if authorize == None:
        raise HTTPException(status_code=403, detail="Not authorized")
    elif isinstance(authorize, AuthJWT):
        user = get_optional_user(authorize)
        require_user_has_agent_access(agent_id, require_user(user))
    else:
        agent = agent_with_id(agent_id)
        if not _contains_access_key(agent, authorize):
            raise HTTPException(status_code=403, detail="Not authorized")
    return True 

