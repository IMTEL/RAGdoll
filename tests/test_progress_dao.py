"""Durable progress state: survives restarts, keeps upload status, TTL."""

from datetime import UTC, datetime

import mongomock
import pytest
from fastapi.testclient import TestClient

from src.config import Config
from src.main import app
from src.models.chat.command import Command
from src.rag_service.dao.progress import mongodb_progress_dao
from src.rag_service.dao.progress.mongodb_progress_dao import MongoDBProgressDAO
from src.routes import chat as chat_routes
from src.routes import progress as progress_routes


@pytest.fixture
def shared_client(monkeypatch):
    client = mongomock.MongoClient()
    monkeypatch.setattr(mongodb_progress_dao, "MongoClient", lambda *a, **k: client)
    return client


def _entry(**overrides):
    entry = {
        "task_name": "t",
        "description": "d",
        "status": "started",
        "agent_id": "a1",
        "session_id": "s1",
        "subtask_progress": [],
        "started_at": datetime.now(UTC),
        "completed_at": None,
    }
    entry.update(overrides)
    return entry


def test_progress_survives_restart(shared_client):
    MongoDBProgressDAO().add_progress(_entry())

    restarted = MongoDBProgressDAO()  # new instance, same database

    [loaded] = restarted.list_session_progress("a1", "s1")
    assert loaded["task_name"] == "t"
    assert loaded["updated_at"].tzinfo is not None


def test_upload_task_survives_restart(shared_client):
    MongoDBProgressDAO().create_upload_task(
        {"task_id": "u1", "agent_id": "a1", "status": "queued"}
    )
    assert MongoDBProgressDAO().get_upload_task("u1")["status"] == "queued"


def test_indexes_and_ttl(shared_client):
    dao = MongoDBProgressDAO()
    ttl = int(Config().PROGRESS_TTL_HOURS * 3600)
    assert ttl == 24 * 3600

    info = dao.collection.index_information()
    assert any(
        [k for k, _ in v["key"]] == ["agent_id", "session_id", "updated_at"]
        for v in info.values()
    )
    assert any(v["key"] == [("task_id", 1)] and v.get("unique") for v in info.values())
    assert any(
        v["key"] == [("updated_at", 1)] and v.get("expireAfterSeconds") == ttl
        for v in info.values()
    )
    upload_info = dao.upload_collection.index_information()
    assert any(
        v["key"] == [("updated_at", 1)] and v.get("expireAfterSeconds") == ttl
        for v in upload_info.values()
    )


def test_legacy_completet_at_is_read_as_completed_at(shared_client):
    dao = MongoDBProgressDAO()
    legacy = datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    dao.collection.insert_one(
        {
            **_entry(),
            "task_id": "old",
            "completed_at": None,
            "completet_at": legacy,
            "updated_at": legacy,
        }
    )
    [loaded] = dao.list_session_progress("a1", "s1")
    assert loaded["completed_at"] == legacy.replace(tzinfo=UTC)
    assert "completet_at" not in loaded
    # a legacy-completed task is not "active" any more
    assert dao.update_active_progress("a1", "s1", "t", {"status": "x"}) is None


def test_update_active_ignores_completed(shared_client):
    dao = MongoDBProgressDAO()
    dao.add_progress(_entry(completed_at=datetime.now(UTC)))
    assert dao.update_active_progress("a1", "s1", "t", {"status": "x"}) is None


@pytest.fixture
def key_auth(monkeypatch):
    monkeypatch.setattr(
        progress_routes.agent_dao, "get_agent_by_id", lambda _id: object()
    )
    monkeypatch.setattr(
        progress_routes.access_service, "authenticate", lambda *_a: True
    )


def test_upload_status_stays_available_after_progress_get(
    progress_store, key_auth, monkeypatch
):
    progress_store.create_upload_task(
        {
            "task_id": "u1",
            "agent_id": "a1",
            "status": "processing",
            "started_at": None,
            "completed_at": None,
        }
    )
    client = TestClient(app)
    for _ in range(2):
        assert (
            client.get(
                "/api/progress", params={"agent_id": "a1"}, headers={"access-key": "k"}
            ).status_code
            == 200
        )
    assert progress_store.get_upload_task("u1")["status"] == "processing"


def test_default_session_does_not_crash_chat(progress_store, key_auth):
    # An upload task (no description/session) used to sit in the same list as
    # progress and broke ProgressData validation for the "default" session.
    progress_store.create_upload_task(
        {"task_id": "u1", "agent_id": "a1", "task_name": "upload_x.pdf"}
    )
    progress_store.add_progress(_entry(session_id="default"))
    command = Command(
        agent_id="a1",
        session_id="default",
        include_progress=True,
        progress_limit=5,
    )

    chat_routes._attach_recent_progress(command)

    assert [t.task_name for t in command.progress] == ["t"]


def test_started_update_then_complete_roundtrip(progress_store, key_auth):
    client = TestClient(app)
    body = {
        "task_name": "t",
        "description": "d",
        "agent_id": "a1",
        "access_key": "k",
        "session_id": "s1",
    }
    first = client.post("/api/progress/updateTask", json={**body, "status": "started"})
    assert first.json()["message"] == "Progress received"
    second = client.post("/api/progress/updateTask", json={**body, "status": "started"})
    assert second.json()["message"] == "Progress updated"
    done = client.post("/api/progress/updateTask", json={**body, "status": "complete"})
    assert done.json()["message"] == "Task completed"
    assert "completet_at" not in done.json()["data"]
    assert done.json()["data"]["completed_at"] is not None
    assert len(progress_store.list_session_progress("a1", "s1")) == 1
