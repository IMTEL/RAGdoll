"""In-process concurrency of the FAISS store (chat threads vs. upload tasks)."""

import threading

import numpy as np
import pytest

from src.rag_service.dao.context import faiss_vector_store
from src.rag_service.dao.context.faiss_vector_store import (
    FAISSIndexError,
    FAISSVectorStore,
    FAISSVectorStoreManager,
)


DIM = 8
AGENT = "agent-race"


def _vec(seed: int) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(DIM)
    return (v / np.linalg.norm(v)).tolist()


def _corrupt_files():
    return [
        p for p in faiss_vector_store.FAISS_INDEX_DIR.iterdir() if "corrupt" in p.name
    ]


@pytest.fixture
def seeded():
    # Seed before the pause fixture is armed.
    manager = FAISSVectorStoreManager()
    manager.add_vector(AGENT, "c0", _vec(0), "doc-1", "n", "t0")
    return manager


@pytest.fixture
def paused_metadata_write(seeded, monkeypatch):
    """Make the next metadata write pause (after the index file was replaced)."""
    real = faiss_vector_store._save_metadata
    paused = threading.Event()
    release = threading.Event()
    state = {"armed": True}

    def slow_save(path, metadata):
        if state["armed"]:
            state["armed"] = False
            paused.set()
            release.wait(10)
        real(path, metadata)

    monkeypatch.setattr(faiss_vector_store, "_save_metadata", slow_save)
    return paused, release


def _run_with_paused_writer(paused, release, writer, other):
    """Run ``other`` while ``writer`` sits between its two file writes.

    Then let the writer finish. Returns the errors raised per thread.
    """
    errors = {}

    def wrap(name, fn):
        def run():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - recorded for the assertion
                errors[name] = e

        return threading.Thread(target=run)

    t_writer = wrap("writer", writer)
    t_writer.start()
    assert paused.wait(10)
    t_other = wrap("other", other)
    t_other.start()
    # With a lock the other thread blocks until the writer is done; without
    # one it finishes on its own while the files disagree.
    t_other.join(1.0)
    release.set()
    t_writer.join(10)
    t_other.join(10)
    return errors


def test_search_while_saving_sees_consistent_store(seeded, paused_metadata_write):
    paused, release = paused_metadata_write
    manager = seeded
    results = {}

    def writer():
        manager.add_vector(AGENT, "c1", _vec(1), "doc-1", "n", "t1")

    def searcher():
        results["hits"] = manager.search(AGENT, _vec(0), ["doc-1"])

    errors = _run_with_paused_writer(paused, release, writer, searcher)

    assert errors == {}
    assert _corrupt_files() == []
    assert "c0" in results["hits"]
    # A fresh load from disk sees both chunks.
    fresh = FAISSVectorStore(AGENT)
    assert set(fresh.chunk_metadata) == {"c0", "c1"}


def test_concurrent_adds_get_unique_ids(seeded, paused_metadata_write):
    paused, release = paused_metadata_write
    manager = seeded

    errors = _run_with_paused_writer(
        paused,
        release,
        lambda: manager.add_vector(AGENT, "c1", _vec(1), "doc-1", "n", "t1"),
        lambda: manager.add_vector(AGENT, "c2", _vec(2), "doc-1", "n", "t2"),
    )

    assert errors == {}
    assert _corrupt_files() == []
    fresh = FAISSVectorStore(AGENT)
    assert set(fresh.chunk_metadata) == {"c0", "c1", "c2"}
    ids = [m["int_id"] for m in fresh.chunk_metadata.values()]
    assert len(set(ids)) == 3
    assert fresh.size() == 3


def _fail_metadata_write(monkeypatch):
    """Fail metadata writes until the returned switch is turned off."""
    real = faiss_vector_store._save_metadata
    failing = {"on": True}

    def maybe_fail(path, metadata):
        if failing["on"]:
            raise OSError("disk full")
        real(path, metadata)

    monkeypatch.setattr(faiss_vector_store, "_save_metadata", maybe_fail)
    return failing


def test_failed_add_save_leaves_memory_matching_disk(seeded, monkeypatch):
    manager = seeded
    failing = _fail_metadata_write(monkeypatch)
    with pytest.raises(FAISSIndexError):
        manager.add_vector(AGENT, "c1", _vec(1), "doc-1", "n", "t1")
    failing["on"] = False

    store = manager.get_store(AGENT)
    assert set(store.chunk_metadata) == {"c0"}
    assert store.size() == 1
    assert _corrupt_files() == []
    # The next add works and the disk stays loadable.
    manager.add_vector(AGENT, "c1", _vec(1), "doc-1", "n", "t1")
    assert set(FAISSVectorStore(AGENT).chunk_metadata) == {"c0", "c1"}


def test_failed_remove_save_leaves_memory_matching_disk(seeded, monkeypatch):
    manager = seeded
    failing = _fail_metadata_write(monkeypatch)
    with pytest.raises(FAISSIndexError):
        manager.remove_chunks(AGENT, ["c0"])
    failing["on"] = False

    store = manager.get_store(AGENT)
    assert set(store.chunk_metadata) == {"c0"}
    assert store.size() == 1
    assert "c0" in manager.search(AGENT, _vec(0), ["doc-1"])
    assert _corrupt_files() == []
    assert set(FAISSVectorStore(AGENT).chunk_metadata) == {"c0"}


def test_failed_first_add_leaves_no_files(monkeypatch):
    manager = FAISSVectorStoreManager()
    failing = _fail_metadata_write(monkeypatch)
    with pytest.raises(FAISSIndexError):
        manager.add_vector("agent-new", "c0", _vec(0), "doc-1", "n", "t0")
    failing["on"] = False

    assert list(faiss_vector_store.FAISS_INDEX_DIR.glob("agent_agent-new*")) == []
    assert manager.get_store("agent-new").size() == 0
    manager.add_vector("agent-new", "c0", _vec(0), "doc-1", "n", "t0")
    assert set(FAISSVectorStore("agent-new").chunk_metadata) == {"c0"}
