"""Mock implementation of ProfileDAO for testing."""

import logging
import uuid
from datetime import datetime

from src.models.profiles import JobSeekerProfile
from src.rag_service.dao.profile.base import EDITABLE_FIELDS, ProfileDAO
from src.utils import singleton


logger = logging.getLogger(__name__)


@singleton
class MockProfileDAO(ProfileDAO):
    """In-memory implementation of ProfileDAO for testing.

    Stores profiles in a dictionary with singleton pattern to
    maintain state across test cases within a session.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, JobSeekerProfile] = {}

    def get_by_id(self, profile_id: str) -> JobSeekerProfile | None:
        """Fetch a profile by its unique ID.

        Args:
            profile_id (str): Unique identifier for the profile

        Returns:
            JobSeekerProfile | None: Profile if found, None otherwise
        """
        profile = self._profiles.get(profile_id)
        return profile.model_copy(deep=True) if profile else None

    def get_by_owner(self, owner_id: str) -> list[JobSeekerProfile]:
        """Fetch all profiles owned by a user, most recently updated first.

        Args:
            owner_id (str): User identifier

        Returns:
            list[JobSeekerProfile]: Profiles owned by the user
        """
        owned = [p for p in self._profiles.values() if p.owner_id == owner_id]
        owned.sort(key=lambda p: p.updated_at, reverse=True)
        return [p.model_copy(deep=True) for p in owned]

    def create(self, profile: JobSeekerProfile) -> JobSeekerProfile:
        """Create a new profile.

        Args:
            profile (JobSeekerProfile): Profile to create

        Returns:
            JobSeekerProfile: Created profile with ID and timestamps populated

        Raises:
            ValueError: If required fields are missing
        """
        if not profile.owner_id:
            raise ValueError("Owner ID is required")

        if not profile.id:
            profile.id = str(uuid.uuid4())

        now = datetime.now()
        profile.created_at = now
        profile.updated_at = now

        # Store a copy to avoid external mutations
        self._profiles[profile.id] = profile.model_copy(deep=True)

        logger.info(
            f"Mock: Created profile '{profile.id}' for owner '{profile.owner_id}'"
        )
        return profile

    def update(self, profile: JobSeekerProfile) -> JobSeekerProfile:
        """Update the editable fields of an existing profile.

        ``owner_id`` and ``created_at`` are never changed.

        Args:
            profile (JobSeekerProfile): Profile with updated fields

        Returns:
            JobSeekerProfile: Updated profile as stored

        Raises:
            ValueError: If the profile does not exist
        """
        if not profile.id:
            raise ValueError("Profile ID is required for update")

        stored = self._profiles.get(profile.id)
        if not stored:
            raise ValueError(f"Profile with ID '{profile.id}' not found")

        changes = profile.model_dump(include=set(EDITABLE_FIELDS))
        changes["updated_at"] = datetime.now()
        updated = stored.model_copy(update=changes, deep=True)
        self._profiles[profile.id] = updated

        logger.info(f"Mock: Updated profile '{profile.id}'")
        return updated.model_copy(deep=True)

    def delete(self, profile_id: str) -> bool:
        """Delete a profile.

        Args:
            profile_id (str): ID of the profile to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            logger.info(f"Mock: Deleted profile '{profile_id}'")
            return True
        return False

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: Always True for mock implementation
        """
        return True

    def clear(self):
        """Clear all profiles from mock storage.

        Useful for test teardown to ensure clean state.
        """
        self._profiles.clear()
        logger.info("Mock: Cleared all profiles")
