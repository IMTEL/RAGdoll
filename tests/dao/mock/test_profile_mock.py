"""Tests for MockProfileDAO implementation."""

import time

import pytest

from src.models.profiles import JobSeekerProfile
from tests.mocks.mock_profile_dao import MockProfileDAO


def _profile(owner_id: str = "user-1", **fields) -> JobSeekerProfile:
    return JobSeekerProfile(owner_id=owner_id, label=fields.pop("label", "A"), **fields)


class TestMockProfileDAO:
    """Tests for MockProfileDAO (in-memory implementation)."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for mock tests."""
        self.repo = MockProfileDAO()
        self.repo.clear()
        yield
        self.repo.clear()

    def test_is_reachable(self):
        """Test that DAO is always reachable."""
        assert self.repo.is_reachable() is True

    def test_create_sets_id_and_timestamps(self):
        """Test that create populates the ID and timestamps."""
        created = self.repo.create(_profile(age_range="30-39"))

        assert created.id is not None
        assert created.age_range == "30-39"
        assert created.created_at == created.updated_at

    def test_create_requires_owner(self):
        """Test that a profile without owner is rejected."""
        with pytest.raises(ValueError):
            self.repo.create(_profile(owner_id=""))

    def test_get_by_id(self):
        """Test fetching a profile by ID."""
        created = self.repo.create(_profile())

        assert self.repo.get_by_id(created.id) == created
        assert self.repo.get_by_id("missing") is None

    def test_get_by_owner_filters_and_sorts(self):
        """Test listing only the owner's profiles, newest first."""
        first = self.repo.create(_profile(label="First"))
        time.sleep(0.01)
        second = self.repo.create(_profile(label="Second"))
        self.repo.create(_profile(owner_id="user-2", label="Other"))

        listed = self.repo.get_by_owner("user-1")

        assert [p.id for p in listed] == [second.id, first.id]

    def test_update_changes_only_editable_fields(self):
        """Test that update never changes owner_id or created_at."""
        created = self.repo.create(_profile())
        edit = created.model_copy(
            update={"label": "Edited", "notes": "note", "owner_id": "user-2"}
        )

        updated = self.repo.update(edit)

        assert updated.label == "Edited"
        assert updated.notes == "note"
        assert updated.owner_id == "user-1"
        assert updated.created_at == created.created_at
        assert updated.updated_at >= created.updated_at

    def test_update_missing_profile(self):
        """Test that updating a missing profile raises error."""
        with pytest.raises(ValueError):
            self.repo.update(_profile().model_copy(update={"id": "missing"}))

    def test_delete(self):
        """Test deleting a profile."""
        created = self.repo.create(_profile())

        assert self.repo.delete(created.id) is True
        assert self.repo.get_by_id(created.id) is None
        assert self.repo.delete(created.id) is False

    def test_returned_profiles_are_copies(self):
        """Test that changing a returned profile does not change storage."""
        created = self.repo.create(_profile())

        fetched = self.repo.get_by_id(created.id)
        fetched.label = "Changed outside the DAO"

        assert self.repo.get_by_id(created.id).label == "A"
