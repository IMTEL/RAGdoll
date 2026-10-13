"""CRUD endpoints for anonymous job seeker profiles.

A profile belongs to the user who created it. Every endpoint only sees the
authenticated user's own profiles: a profile owned by someone else is
reported as 404, so its existence is not revealed.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_jwt_auth import AuthJWT
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.models.profiles import AgeRange, ExperienceLevel, JobSeekerProfile
from src.models.users.user import User
from src.rag_service.dao.factory import get_profile_dao
from src.rag_service.dao.profile.base import ProfileDAO
from src.routes.agents import _get_user_or_demo, optional_auth


router = APIRouter(prefix="/profiles", tags=["profiles"])

Language = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)
]
PracticeArea = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class ProfileRequest(BaseModel):
    """Body for creating or replacing a profile.

    Only the editable fields are accepted. ``id``, ``owner_id`` and the
    timestamps are set by the server, and unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = Field(..., min_length=1, max_length=100)
    age_range: AgeRange | None = None
    experience_level: ExperienceLevel | None = None
    education: str = Field(default="", max_length=500)
    work_experience: str = Field(default="", max_length=1000)
    languages: list[Language] = Field(default_factory=list, max_length=20)
    practice_areas: list[PracticeArea] = Field(default_factory=list, max_length=20)
    notes: str = Field(default="", max_length=2000)


@lru_cache
def get_dao() -> ProfileDAO:
    """Return the profile DAO, created once per process."""
    return get_profile_dao()


def get_current_user(
    authorize: Annotated[AuthJWT | None, Depends(optional_auth)] = None,
) -> User:
    """Return the authenticated user, or the demo user if auth is disabled."""
    return _get_user_or_demo(authorize)


CurrentUser = Annotated[User, Depends(get_current_user)]
Dao = Annotated[ProfileDAO, Depends(get_dao)]


def _get_owned_profile(
    dao: ProfileDAO, profile_id: str, user: User
) -> JobSeekerProfile:
    """Fetch a profile owned by the user, or raise 404."""
    profile = dao.get_by_id(profile_id)
    if profile is None or profile.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


@router.get("", response_model=list[JobSeekerProfile])
def list_profiles(user: CurrentUser, dao: Dao):
    """List the user's profiles, most recently updated first."""
    return dao.get_by_owner(user.id)


@router.post("", response_model=JobSeekerProfile, status_code=status.HTTP_201_CREATED)
def create_profile(payload: ProfileRequest, user: CurrentUser, dao: Dao):
    """Create a profile owned by the user."""
    profile = JobSeekerProfile(owner_id=user.id, **payload.model_dump())
    return dao.create(profile)


@router.get("/{profile_id}", response_model=JobSeekerProfile)
def get_profile(profile_id: str, user: CurrentUser, dao: Dao):
    """Return one of the user's profiles."""
    return _get_owned_profile(dao, profile_id, user)


@router.put("/{profile_id}", response_model=JobSeekerProfile)
def update_profile(
    profile_id: str, payload: ProfileRequest, user: CurrentUser, dao: Dao
):
    """Replace the editable fields of one of the user's profiles."""
    profile = _get_owned_profile(dao, profile_id, user)
    updated = profile.model_copy(update=payload.model_dump())
    try:
        return dao.update(updated)
    except ValueError as e:
        # Deleted between the read and the update
        raise HTTPException(status_code=404, detail="Profile not found") from e


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_profile(profile_id: str, user: CurrentUser, dao: Dao):
    """Delete one of the user's profiles."""
    _get_owned_profile(dao, profile_id, user)
    if not dao.delete(profile_id):
        raise HTTPException(status_code=404, detail="Profile not found")
