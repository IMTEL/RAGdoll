from typing import Annotated, Literal

from fastapi import Depends, HTTPException, Request
from fastapi_jwt_auth import AuthJWT

from src.globals import auth_service, agent_dao
from src.models.agent import Agent
from src.models.users.user import User

def get_auth(request: Request) -> AuthJWT | None:
    """Return AuthJWT or None depending if a token is accessible."""
    if not request.headers.get("authorization"):
        # Passing none, letting the authentication service take care of it
        return None
    return AuthJWT(request)

def current_user(authorize: Annotated[AuthJWT | None, Depends(get_auth)]) -> User:
    """
    Dependency allowing for fetching of the current user
    """
    return auth_service.get_authenticated_user(authorize);

def agent_with_id(agent_id : str) -> Agent:
    agent = agent_dao.get_agent_by_id(agent_id)
    if agent == None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

def has_agent_ownership(agent_id : str, user : Annotated[User, Depends(current_user)]) -> Literal[True]:
    if not agent_id in user.owned_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return True

def has_agent_access(agent_id : str, user : Annotated[User, Depends(current_user)]) -> Literal[True]:
    if not (agent_id in user.collaborating_agents or has_agent_ownership(agent_id, user)):
        raise HTTPException(status_code=404, detail="Agent not found") 
    return True
