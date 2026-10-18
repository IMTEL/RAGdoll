"""MongoDB implementation of ProgressDAO.

Collections:
- ``progress``: training progress. Indexes: ``{agent_id, session_id,
  updated_at}``, unique ``task_id`` and a TTL on ``updated_at``.
- ``upload_tasks``: document upload status. Unique ``task_id`` and the same TTL.
"""

import logging
import uuid
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument

from src.config import Config
from src.rag_service.dao.progress.base import ProgressDAO, normalize_entry, utc_now


logger = logging.getLogger(__name__)

config = Config()


class MongoDBProgressDAO(ProgressDAO):
    def __init__(self):
        self.client = MongoClient(config.MONGODB_URI)
        self.db = self.client[config.MONGODB_DATABASE]
        self.collection = self.db[config.MONGODB_PROGRESS_COLLECTION]
        self.upload_collection = self.db[config.MONGODB_UPLOAD_TASKS_COLLECTION]
        self._create_indexes()

    def _create_indexes(self):
        ttl = int(config.PROGRESS_TTL_HOURS * 3600)
        try:
            self.collection.create_index(
                [
                    ("agent_id", ASCENDING),
                    ("session_id", ASCENDING),
                    ("updated_at", DESCENDING),
                ]
            )
            self.collection.create_index([("task_id", ASCENDING)], unique=True)
            self.collection.create_index(
                [("updated_at", ASCENDING)], expireAfterSeconds=ttl
            )
            self.upload_collection.create_index([("task_id", ASCENDING)], unique=True)
            self.upload_collection.create_index(
                [("updated_at", ASCENDING)], expireAfterSeconds=ttl
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not create progress indexes: %s", type(e).__name__)

    # --- training progress -------------------------------------------------

    def add_progress(self, entry: dict[str, Any]) -> dict[str, Any]:
        doc = dict(entry)
        doc.pop("_id", None)
        doc.setdefault("task_id", str(uuid.uuid4()))
        doc["updated_at"] = doc.get("updated_at") or utc_now()
        self.collection.insert_one(doc)
        return normalize_entry(doc)

    def update_active_progress(
        self, agent_id: str, session_id: str, task_name: str, changes: dict[str, Any]
    ) -> dict[str, Any] | None:
        updated = self.collection.find_one_and_update(
            {
                "agent_id": agent_id,
                "session_id": session_id,
                "task_name": task_name,
                "completed_at": None,
                "completet_at": None,
            },
            {"$set": changes},
            return_document=ReturnDocument.AFTER,
        )
        return normalize_entry(updated) if updated else None

    def list_session_progress(
        self, agent_id: str, session_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find(
            {"agent_id": agent_id, "session_id": session_id}
        ).sort("updated_at", DESCENDING)
        if limit is not None and limit >= 0:
            cursor = cursor.limit(limit)
        return [normalize_entry(doc) for doc in cursor]

    def list_progress_for_agents(self, agent_ids: set[str]) -> list[dict[str, Any]]:
        query = {"agent_id": {"$in": sorted(agent_ids)}}
        docs = list(self.collection.find(query).sort("updated_at", ASCENDING))
        docs += list(self.upload_collection.find(query).sort("updated_at", ASCENDING))
        return [normalize_entry(doc) for doc in docs]

    # --- upload tasks ------------------------------------------------------

    def create_upload_task(self, entry: dict[str, Any]) -> dict[str, Any]:
        doc = dict(entry)
        doc.pop("_id", None)
        doc["updated_at"] = doc.get("updated_at") or utc_now()
        self.upload_collection.insert_one(doc)
        return normalize_entry(doc)

    def update_upload_task(self, task_id: str, changes: dict[str, Any]) -> None:
        self.upload_collection.update_one(
            {"task_id": task_id}, {"$set": {**changes, "updated_at": utc_now()}}
        )

    def get_upload_task(self, task_id: str) -> dict[str, Any] | None:
        doc = self.upload_collection.find_one({"task_id": task_id})
        return normalize_entry(doc) if doc else None
