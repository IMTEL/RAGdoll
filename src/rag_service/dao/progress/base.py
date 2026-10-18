"""Abstract interface for training-progress and upload-status storage."""

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

from src.models.accesskey import as_utc


_DATETIME_FIELDS = ("started_at", "completed_at", "updated_at", "created_at")


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a stored entry in its public shape.

    Drops storage-internal keys, returns datetimes as aware UTC and reads the
    legacy ``completet_at`` spelling into ``completed_at`` for old entries.
    """
    cleaned = {k: v for k, v in entry.items() if k != "_id"}
    legacy = cleaned.pop("completet_at", None)
    if cleaned.get("completed_at") is None and legacy is not None:
        cleaned["completed_at"] = legacy
    for field in _DATETIME_FIELDS:
        value = cleaned.get(field)
        if isinstance(value, datetime):
            cleaned[field] = as_utc(value)
    return cleaned


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProgressDAO(ABC):
    """Durable storage for session progress and document-upload status."""

    # --- training progress -------------------------------------------------

    @abstractmethod
    def add_progress(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert a progress entry (a ``task_id`` is generated when missing)."""

    @abstractmethod
    def update_active_progress(
        self, agent_id: str, session_id: str, task_name: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Apply ``changes`` to the open (not completed) task with that name.

        Returns the updated entry, or None when no open task matches.
        """

    @abstractmethod
    def list_session_progress(
        self, agent_id: str, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Entries of one session, newest ``updated_at`` first."""

    @abstractmethod
    def list_progress_for_agents(self, agent_ids: set[str]) -> list[dict[str, Any]]:
        """All progress entries and upload tasks belonging to ``agent_ids``."""

    # --- upload tasks ------------------------------------------------------

    @abstractmethod
    def create_upload_task(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Insert an upload task (must contain ``task_id``)."""

    @abstractmethod
    def update_upload_task(self, task_id: str, changes: dict[str, Any]) -> None:
        """Merge ``changes`` into an upload task and refresh ``updated_at``."""

    @abstractmethod
    def get_upload_task(self, task_id: str) -> dict[str, Any] | None:
        """Fetch an upload task by id."""
