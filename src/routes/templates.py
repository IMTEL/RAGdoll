"""CRUD endpoints for templates that users reuse when configuring agents.

A template belongs to the user who created it. Every endpoint only sees the
authenticated user's own templates: a template owned by someone else is
reported as 404, so its existence is not revealed.
"""

import json
from datetime import UTC, datetime
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.template import Template
from src.models.users.user import User
from src.rag_service.dao.factory import get_template_dao
from src.rag_service.dao.template.base import TemplateDAO
from src.routes.agents import _get_user_or_demo, optional_auth


router = APIRouter(prefix="/templates", tags=["templates"])

MAX_CONTENT_BYTES = 64 * 1024


class TemplateRequest(BaseModel):
    """Body for creating or replacing a template.

    Only the editable fields are accepted. ``id``, ``owner_id`` and the
    timestamps are set by the server, and unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: str = Field(..., min_length=1, max_length=64)
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=1000)
    content: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _limit_content_size(cls, content: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(content).encode()) > MAX_CONTENT_BYTES:
            raise ValueError(f"content must be at most {MAX_CONTENT_BYTES} bytes")
        return content


@lru_cache
def get_dao() -> TemplateDAO:
    """Return the template DAO, created once per process."""
    return get_template_dao()


def get_current_user(
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
) -> User:
    """Return the authenticated user, or the demo user if auth is disabled."""
    return _get_user_or_demo(authorize)


CurrentUser = Annotated[User, Depends(get_current_user)]
Dao = Annotated[TemplateDAO, Depends(get_dao)]


def _get_owned_template(dao: TemplateDAO, template_id: str, user: User) -> Template:
    """Fetch a template owned by the user, or raise 404."""
    template = dao.get_by_id(template_id)
    if template is None or template.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Template not found")
    return template


@router.get("", response_model=list[Template])
def list_templates(
    user: CurrentUser,
    dao: Dao,
    template_type: Annotated[str | None, Query(alias="type")] = None,
):
    """List the user's templates, newest first, optionally of one type only."""
    return dao.get_by_owner(user.id, template_type)


@router.post("", response_model=Template, status_code=status.HTTP_201_CREATED)
def create_template(payload: TemplateRequest, user: CurrentUser, dao: Dao):
    """Create a template owned by the user."""
    return dao.create(Template(owner_id=user.id, **payload.model_dump()))


@router.get("/{template_id}", response_model=Template)
def get_template(template_id: str, user: CurrentUser, dao: Dao):
    """Return one of the user's templates."""
    return _get_owned_template(dao, template_id, user)


@router.put("/{template_id}", response_model=Template)
def update_template(
    template_id: str, payload: TemplateRequest, user: CurrentUser, dao: Dao
):
    """Replace the editable fields of one of the user's templates."""
    template = _get_owned_template(dao, template_id, user)
    updated = template.model_copy(
        update={**payload.model_dump(), "updated_at": datetime.now(UTC)}
    )
    try:
        return dao.update(updated)
    except ValueError as e:
        # Deleted between the read and the update
        raise HTTPException(status_code=404, detail="Template not found") from e


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_template(template_id: str, user: CurrentUser, dao: Dao):
    """Delete one of the user's templates."""
    _get_owned_template(dao, template_id, user)
    if not dao.delete(template_id):
        raise HTTPException(status_code=404, detail="Template not found")
