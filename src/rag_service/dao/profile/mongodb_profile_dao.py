"""MongoDB implementation for job seeker profile storage."""

import logging
import uuid
from datetime import datetime

from pymongo import ASCENDING, DESCENDING, MongoClient

from src.config import Config
from src.models.profiles import JobSeekerProfile
from src.rag_service.dao.profile.base import EDITABLE_FIELDS, ProfileDAO


logger = logging.getLogger(__name__)

config = Config()


class MongoDBProfileDAO(ProfileDAO):
    """MongoDB-backed data access object for job seeker profiles."""

    def __init__(self):
        """Initialize MongoDB connection and create indexes."""
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_PROFILES_COLLECTION]

        self._create_indexes()

    def _create_indexes(self):
        """Create database indexes for optimized queries."""
        try:
            # Listing a user's profiles is the most common query
            self.collection.create_index(
                [("owner_id", ASCENDING), ("updated_at", DESCENDING)]
            )
            logger.info("Profile collection indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")

    def get_by_id(self, profile_id: str) -> JobSeekerProfile | None:
        """Fetch a profile by its unique ID.

        Args:
            profile_id (str): Unique identifier for the profile

        Returns:
            JobSeekerProfile | None: Profile if found, None otherwise
        """
        if not profile_id:
            return None

        doc = self.collection.find_one({"_id": profile_id})
        if not doc:
            return None

        return self._profile_from_mongo(doc)

    def get_by_owner(self, owner_id: str) -> list[JobSeekerProfile]:
        """Fetch all profiles owned by a user, most recently updated first.

        Args:
            owner_id (str): User identifier

        Returns:
            list[JobSeekerProfile]: Profiles owned by the user
        """
        if not owner_id:
            return []

        cursor = self.collection.find({"owner_id": owner_id}).sort(
            "updated_at", DESCENDING
        )
        return [self._profile_from_mongo(doc) for doc in cursor]

    def create(self, profile: JobSeekerProfile) -> JobSeekerProfile:
        """Create a new profile.

        Args:
            profile (JobSeekerProfile): Profile to create

        Returns:
            JobSeekerProfile: Created profile with ID and timestamps populated

        Raises:
            ValueError: If required fields are missing or the insert fails
        """
        if not profile.owner_id:
            raise ValueError("Owner ID is required")

        if not profile.id:
            profile.id = str(uuid.uuid4())

        now = datetime.now()
        profile.created_at = now
        profile.updated_at = now

        doc = profile.model_dump(exclude={"id"})
        doc["_id"] = profile.id

        try:
            self.collection.insert_one(doc)
            logger.info(
                f"Created profile '{profile.id}' for owner '{profile.owner_id}'"
            )
            return profile
        except Exception as e:
            logger.error(f"Failed to create profile: {e}")
            raise ValueError(f"Failed to create profile: {e}") from e

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

        changes = profile.model_dump(include=set(EDITABLE_FIELDS))
        changes["updated_at"] = datetime.now()

        try:
            result = self.collection.update_one({"_id": profile.id}, {"$set": changes})
        except Exception as e:
            logger.error(f"Failed to update profile: {e}")
            raise ValueError(f"Failed to update profile: {e}") from e

        if result.matched_count == 0:
            raise ValueError(f"Profile with ID '{profile.id}' not found")

        logger.info(f"Updated profile '{profile.id}'")
        return self.get_by_id(profile.id)

    def delete(self, profile_id: str) -> bool:
        """Delete a profile.

        Args:
            profile_id (str): ID of the profile to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not profile_id:
            return False

        try:
            result = self.collection.delete_one({"_id": profile_id})
        except Exception as e:
            logger.error(f"Failed to delete profile: {e}")
            return False

        if result.deleted_count > 0:
            logger.info(f"Deleted profile '{profile_id}'")
            return True

        logger.warning(f"Profile '{profile_id}' not found for deletion")
        return False

    def is_reachable(self) -> bool:
        """Check if the DAO backend is accessible.

        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            self.client.admin.command("ping")
            logger.debug("Successfully pinged MongoDB")
            return True
        except Exception as e:
            logger.error(f"Failed to ping MongoDB: {e}")
            return False

    def _profile_from_mongo(self, doc: dict) -> JobSeekerProfile:
        """Convert a MongoDB document to a JobSeekerProfile model.

        Args:
            doc (dict): MongoDB document

        Returns:
            JobSeekerProfile: Profile model instance
        """
        data = {key: value for key, value in doc.items() if key != "_id"}
        return JobSeekerProfile(id=str(doc["_id"]), **data)
