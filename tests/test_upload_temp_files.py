import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import context_upload
from src.main import app
from src.models.agent import Agent
from src.routes import upload
from src.utils.path_dependencies import get_auth


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    directory = tmp_path / "temp_files"
    monkeypatch.setattr(upload, "TEMP_FILES_DIR", directory)
    return directory


@pytest.fixture
def client(monkeypatch):
    agent = Agent(
        name="a",
        description="d",
        prompt="p",
        llm_model="gpt-4",
        llm_api_key="k",
        embedding_api_key="k",
        embedding_model="stub:model",
        last_updated="2026-10-18T00:00:00Z",
    )

    class _AgentDao:
        def get_agent_by_id(self, agent_id):
            return agent

    monkeypatch.setattr(upload, "agent_dao", _AgentDao())
    app.dependency_overrides[get_auth] = lambda: None
    yield TestClient(app)
    app.dependency_overrides.pop(get_auth, None)


def _post(client, content: bytes, name: str = "x.txt"):
    return client.post(
        "/upload/agent",
        params={"agent_id": "agent-1"},
        files={"file": (name, content)},
    )


def test_same_name_uploads_get_distinct_temp_files(client, temp_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(
        upload,
        "_process_document_background",
        lambda **kw: seen.append(kw["file_location"]),
    )
    assert _post(client, b"first").status_code == 200
    assert _post(client, b"second").status_code == 200

    assert len(set(seen)) == 2
    assert sorted(p.read_text() for p in seen) == ["first", "second"]
    assert all(p.name != "x.txt" for p in seen)


def test_background_passes_client_filename_and_cleans_up(client, temp_dir, monkeypatch):
    calls = []

    def fake_process(path, agent_id, model, **kwargs):
        calls.append((Path(path), Path(path).read_text(), kwargs))
        return True, "doc-1"

    monkeypatch.setattr(upload, "process_file_and_store", fake_process)
    assert _post(client, b"hello", "../../evil/x.txt").status_code == 200

    path, content, kwargs = calls[0]
    assert content == "hello"
    assert kwargs["document_name"] == "x.txt"
    assert path.name != "x.txt"
    assert list(temp_dir.iterdir()) == []


def test_oversize_upload_returns_413_and_leaves_no_file(client, temp_dir, monkeypatch):
    monkeypatch.setattr(upload.config, "MAX_UPLOAD_MB", 1, raising=False)
    response = _post(client, b"a" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert list(temp_dir.iterdir()) == []


def test_sweep_removes_only_stale_files(tmp_path):
    old = tmp_path / "old.txt"
    new = tmp_path / "new.txt"
    old.write_text("o")
    new.write_text("n")
    stale = time.time() - 2 * 3600
    import os

    os.utime(old, (stale, stale))

    upload.sweep_stale_temp_files(tmp_path, max_age_seconds=3600)
    assert not old.exists()
    assert new.exists()


def test_sweep_ignores_missing_directory(tmp_path):
    upload.sweep_stale_temp_files(tmp_path / "nope")


def test_document_lock_is_per_agent_and_name():
    a = context_upload._document_lock("ag", "x.txt")
    assert a is context_upload._document_lock("ag", "x.txt")
    assert a is not context_upload._document_lock("ag", "y.txt")
    assert a is not context_upload._document_lock("other", "x.txt")


def test_same_document_processing_is_serialized(tmp_path, monkeypatch):
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def slow_scrape(_path):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.05)
        with guard:
            state["active"] -= 1
        return []

    monkeypatch.setattr(context_upload.scraper_service, "scrape_file", slow_scrape)
    monkeypatch.setattr(context_upload, "get_document_dao", lambda: object())
    monkeypatch.setattr(context_upload, "get_context_dao", lambda: object())
    files = []
    for i in range(3):
        p = tmp_path / f"t{i}.txt"
        p.write_text("x")
        files.append(p)

    threads = [
        threading.Thread(
            target=context_upload.process_file_and_store,
            args=(str(p), "ag", "stub:model"),
            kwargs={"document_name": "same.txt"},
        )
        for p in files
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["max"] == 1
