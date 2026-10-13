"""Abstract base class for Job Seeker Profile DAO pattern."""

from abc import ABC, abstractmethod

from src.models.profiles import JobSeekerProfile


# Fields a client may change after creation
EDITABLE_FIELDS = (
    "label",
    "age_range",
    "experience_level",
    "education",
    "work_experience",
    "languages",
    "practice_areas",
    "notes",
)


class ProfileDAO(ABC):
    """Abstract base class for job seeker profile storage.

    The DAO does not check ownership. Callers must compare
    ``profile.owner_id`` with the authenticated user before returning,
    updating or deleting a profile.
    """

    @abstractmethod
    def get_by_id(self, profile_id: str) -> JobSeekerProfile | None:
        """Fetch a profile by its unique ID.

        Args:
            profile_id (str): Unique identifier for the profile

        Returns:
            JobSeekerProfile | None: Profile if found, None otherwise
        """

    @abstractmethod
    def get_by_owner(self, owner_id: str) -> list[JobSeekerProfile]:
        """Fetch all profiles owned by a user, most recently updated first.

        Args:
            owner_id (str): User identifier

        Returns:
            list[JobSeekerProfile]: Profiles owned by the user
        """

    @abstractmethod
    def create(self, profile: JobSeekerProfile) -> JobSeekerProfile:
        """Create a new profile.

        Args:
            profile (JobSeekerProfile): Profile to create

        Returns:
            JobSeekerProfile: Created profile with ID and timestamps populated

        Raises:
            ValueError: If required fields are missing
        """

    @abstractmethod
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

    @abstractmethod
    def delete(self, profile_id: str) -> bool:
        """Delete a profile.

        Args:
            profile_id (str): ID of the profile to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """

    @abstractmethod
    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
