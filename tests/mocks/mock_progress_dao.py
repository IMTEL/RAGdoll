"""In-memory ProgressDAO for tests."""

import copy
import uuid
from typing import Any

from src.rag_service.dao.progress.base import ProgressDAO, normalize_entry, utc_now
from src.utils import singleton


@singleton
class MockProgressDAO(ProgressDAO):
    def __init__(self) -> None:
        self.progress: list[dict[str, Any]] = []
        self.uploads: dict[str, dict[str, Any]] = {}

    def add_progress(self, entry: dict[str, Any]) -> dict[str, Any]:
        doc = copy.deepcopy(entry)
        doc.setdefault("task_id", str(uuid.uuid4()))
        doc["updated_at"] = doc.get("updated_at") or utc_now()
        self.progress.append(doc)
        return normalize_entry(doc)

    def update_active_progress(
        self, agent_id: str, session_id: str, task_name: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        for doc in self.progress:
            if (
                doc.get("agent_id") == agent_id
                and doc.get("session_id") == session_id
                and doc.get("task_name") == task_name
                and doc.get("completed_at") is None
                and doc.get("completet_at") is None
            ):
                doc.update(copy.deepcopy(changes))
                return normalize_entry(doc)
        return None

    def list_session_progress(
        self, agent_id: str, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        docs = [
            d
            for d in self.progress
            if d.get("agent_id") == agent_id and d.get("session_id") == session_id
        ]
        docs.sort(key=lambda d: d["updated_at"], reverse=True)
        if limit is not None and limit >= 0:
            docs = docs[:limit]
        return [normalize_entry(d) for d in docs]

    def list_progress_for_agents(self, agent_ids: set[str]) -> list[dict[str, Any]]:
        docs = [d for d in self.progress if d.get("agent_id") in agent_ids]
        docs += [d for d in self.uploads.values() if d.get("agent_id") in agent_ids]
        return [normalize_entry(d) for d in docs]

    def create_upload_task(self, entry: dict[str, Any]) -> dict[str, Any]:
        doc = copy.deepcopy(entry)
        doc["updated_at"] = doc.get("updated_at") or utc_now()
        self.uploads[doc["task_id"]] = doc
        return normalize_entry(doc)

    def update_upload_task(self, task_id: str, changes: dict[str, Any]) -> None:
        if task_id in self.uploads:
            self.uploads[task_id].update(changes)
            self.uploads[task_id]["updated_at"] = utc_now()

    def get_upload_task(self, task_id: str) -> dict[str, Any] | None:
        doc = self.uploads.get(task_id)
        return normalize_entry(doc) if doc else None
