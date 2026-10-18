import hashlib
import sys

import numpy as np
import pytest

from src import context_upload
from src.rag_service.dao.context import faiss_vector_store
from src.rag_service.dao.context.faiss_vector_store import FAISSVectorStoreManager
from src.rag_service.dao.context.hybrid_search import vector_search
from src.rag_service.dao.factory import get_context_dao, get_document_dao


DIM = 384


def _embed(text: str, *_args, **_kwargs) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vec = np.random.default_rng(seed).standard_normal(DIM)
    return (vec / np.linalg.norm(vec)).tolist()


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(faiss_vector_store, "FAISS_INDEX_DIR", tmp_path / "faiss")
    monkeypatch.setattr(FAISSVectorStoreManager, "_stores", {})
    monkeypatch.setattr(context_upload, "compute_embedding", _embed)
    # mongomock gives every client its own store, so share one DAO instance.
    context_dao = get_context_dao()
    document_dao = get_document_dao()
    monkeypatch.setattr(context_upload, "get_context_dao", lambda: context_dao)
    monkeypatch.setattr(context_upload, "get_document_dao", lambda: document_dao)
    monkeypatch.setattr(sys.modules[__name__], "get_context_dao", lambda: context_dao)
    yield


def _upload(tmp_path, text, agent_id):
    # Same path every time, as with temp files of identically named uploads.
    path = tmp_path / "notes.txt"
    path.write_text(text)
    ok, doc_id = context_upload.process_file_and_store(
        str(path), agent_id, "stub:model"
    )
    assert ok
    return doc_id


def test_same_filename_across_agents_does_not_leak_text(tmp_path):
    text_a = "Agent A secret: the vault code is alpha."
    text_b = "Agent B secret: the vault code is bravo."
    doc_b = _upload(tmp_path, text_b, "agent-b")
    doc_a = _upload(tmp_path, text_a, "agent-a")

    collection = get_context_dao().collection
    ids = [d["chunk_id"] for d in collection.find({})]
    assert len(ids) == len(set(ids)) == 2

    document_map: dict = {}
    scores = vector_search(
        "agent-a", _embed(text_b), [doc_a], collection, 10, 5, document_map
    )

    assert doc_b not in {d["document_id"] for d in document_map.values()}
    assert all(d["agent_id"] == "agent-a" for d in document_map.values())
    assert all("bravo" not in d["text"] for d in document_map.values())
    assert set(scores) >= set(document_map)


def test_vector_search_filters_mongo_lookup_by_agent():
    collection = get_context_dao().collection
    collection.insert_many(
        [
            {"chunk_id": "dup", "agent_id": "agent-b", "text": "B text"},
            {"chunk_id": "dup", "agent_id": "agent-a", "text": "A text"},
        ]
    )
    manager = FAISSVectorStoreManager()
    manager.add_vector("agent-a", "dup", _embed("q"), "doc-a", "n", "A text", 0, 1)

    document_map: dict = {}
    vector_search("agent-a", _embed("q"), ["doc-a"], collection, 10, 5, document_map)

    assert document_map["dup"]["text"] == "A text"


def test_vector_without_mongo_row_is_skipped():
    collection = get_context_dao().collection
    manager = FAISSVectorStoreManager()
    manager.add_vector(
        "agent-a", "orphan", _embed("q"), "doc-a", "n", "stale faiss text", 0, 1
    )

    document_map: dict = {}
    scores = vector_search(
        "agent-a", _embed("q"), ["doc-a"], collection, 10, 5, document_map
    )

    assert document_map == {}
    assert "orphan" not in scores


def test_reupload_yields_new_chunk_ids_without_mongo_duplicates(tmp_path):
    doc_id = _upload(tmp_path, "First version of the text.", "agent-a")
    collection = get_context_dao().collection
    first_ids = {d["chunk_id"] for d in collection.find({"document_id": doc_id})}

    doc_id_2 = _upload(tmp_path, "Second version of the text.", "agent-a")
    rows = list(collection.find({"document_id": doc_id}))
    second_ids = {d["chunk_id"] for d in rows}

    assert doc_id_2 == doc_id
    assert len(rows) == len(second_ids) == len(first_ids)
    assert first_ids.isdisjoint(second_ids)
    assert all("Second" in d["text"] for d in rows)

    # Stale vectors may linger in FAISS for now; they must never surface.
    document_map: dict = {}
    scores = vector_search(
        "agent-a",
        _embed("First version of the text."),
        [doc_id],
        collection,
        10,
        5,
        document_map,
    )
    assert set(document_map) <= second_ids
    assert set(scores) <= second_ids
