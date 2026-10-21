"""Endpoints for a user's settings.

Each client application keeps its settings under its own namespace, so one
client never overwrites another client's settings. RAGdoll does not interpret
the settings.
"""

import json
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, status

from src.models.users.user import User
from src.rag_service.dao.factory import get_user_dao
from src.rag_service.dao.user.base import UserDao
from src.utils.path_dependencies import require_user


router = APIRouter(prefix="/user-settings", tags=["user-settings"])

MAX_SETTINGS_BYTES = 16 * 1024

Namespace = Annotated[str, Path(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]


@lru_cache
def get_dao() -> UserDao:
    """Return the user DAO, created once per process."""
    return get_user_dao()


def get_current_user(user: Annotated[User, Depends(require_user)]) -> User:
    """Return the authenticated user."""
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
Dao = Annotated[UserDao, Depends(get_dao)]


@router.get("/{namespace}", response_model=dict[str, Any])
def get_user_settings(namespace: Namespace, user: CurrentUser):
    """The user's settings for one application. Empty if none are saved."""
    return user.settings.get(namespace, {})


@router.put("/{namespace}", response_model=dict[str, Any])
def put_user_settings(
    namespace: Namespace, settings: dict[str, Any], user: CurrentUser, dao: Dao
):
    """Replace the user's settings for one application."""
    if len(json.dumps(settings).encode()) > MAX_SETTINGS_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Settings are too large",
        )

    user.settings[namespace] = settings
    dao.set_user(user)
    return settings
