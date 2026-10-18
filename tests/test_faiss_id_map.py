import hashlib
import logging
import sys

import faiss
import numpy as np
import pytest

from src import context_upload
from src.models.errors import EmbeddingError
from src.rag_service.dao.context import faiss_vector_store
from src.rag_service.dao.context.faiss_vector_store import (
    CorruptIndexError,
    FAISSVectorStore,
    FAISSVectorStoreManager,
)
from src.rag_service.dao.context.hybrid_search import vector_search
from src.rag_service.dao.factory import get_context_dao, get_document_dao


DIM = 16


def _embed(text: str, *_args, **_kwargs) -> list[float]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vec = np.random.default_rng(seed).standard_normal(DIM)
    return (vec / np.linalg.norm(vec)).tolist()


def _pair_with_cosine(cos: float, dim: int = DIM) -> tuple[list[float], list[float]]:
    u = np.zeros(dim)
    u[0] = 1.0
    v = np.zeros(dim)
    v[0] = cos
    v[1] = np.sqrt(1 - cos**2)
    return u.tolist(), v.tolist()


@pytest.fixture
def daos(monkeypatch):
    monkeypatch.setattr(context_upload, "compute_embedding", _embed)
    # mongomock gives every client its own store, so share one context DAO.
    context_dao = get_context_dao()
    document_dao = get_document_dao()
    document_dao.context_dao = context_dao
    monkeypatch.setattr(context_upload, "get_context_dao", lambda: context_dao)
    monkeypatch.setattr(context_upload, "get_document_dao", lambda: document_dao)
    monkeypatch.setattr(sys.modules[__name__], "get_context_dao", lambda: context_dao)
    return context_dao, document_dao


def _upload(tmp_path, text, agent_id="agent-x", name="notes.txt"):
    path = tmp_path / name
    path.write_text(text)
    return context_upload.process_file_and_store(
        str(path), agent_id, "stub:model", document_name=name
    )


def test_new_index_is_id_mapped_inner_product():
    manager = FAISSVectorStoreManager()
    manager.add_vector("agent-x", "c0", _embed("a"), "doc-1", "n", "t")
    index = manager.get_store("agent-x").index
    assert isinstance(index, faiss.IndexIDMap2)
    assert faiss.downcast_index(index.index).metric_type == faiss.METRIC_INNER_PRODUCT


def test_cosine_score_is_exact():
    u, v = _pair_with_cosine(0.8)
    manager = FAISSVectorStoreManager()
    manager.add_vector("agent-x", "c0", v, "doc-1", "n", "t")
    scores = manager.search("agent-x", u, ["doc-1"])
    assert scores["c0"] == pytest.approx(0.8, abs=1e-5)


def test_negative_cosine_is_kept_in_range():
    u, _ = _pair_with_cosine(0.0)
    manager = FAISSVectorStoreManager()
    manager.add_vector("agent-x", "c0", [-x for x in u], "doc-1", "n", "t")
    scores = manager.search("agent-x", u, ["doc-1"], similarity_threshold=-1.0)
    assert scores["c0"] == pytest.approx(-1.0, abs=1e-5)
    assert scores["c0"] >= -1.0
    # The default threshold of 0 drops opposite vectors.
    assert manager.search("agent-x", u, ["doc-1"]) == {}


def test_remove_document_removes_only_its_vectors_and_persists():
    manager = FAISSVectorStoreManager()
    for i in range(3):
        manager.add_vector("agent-x", f"a{i}", _embed(f"a{i}"), "doc-a", "a", "t")
    for i in range(2):
        manager.add_vector("agent-x", f"b{i}", _embed(f"b{i}"), "doc-b", "b", "t")

    assert manager.remove_document("agent-x", "doc-a") == 3

    FAISSVectorStoreManager._stores.clear()
    store = FAISSVectorStore("agent-x")
    assert store.size() == 2
    assert set(store.chunk_metadata) == {"b0", "b1"}
    scores = manager.search("agent-x", _embed("b1"), ["doc-a", "doc-b"], top_k=5)
    assert next(iter(scores)) == "b1"
    assert scores["b1"] == pytest.approx(1.0, abs=1e-5)
    assert not {"a0", "a1", "a2"} & set(scores)
    # New ids never reuse removed ones.
    manager.add_vector("agent-x", "c0", _embed("c0"), "doc-c", "c", "t")
    assert manager.search("agent-x", _embed("c0"), ["doc-c"], top_k=1) == {
        "c0": pytest.approx(1.0, abs=1e-5)
    }


def test_remove_document_keeps_listed_chunks():
    manager = FAISSVectorStoreManager()
    for i in range(3):
        manager.add_vector("agent-x", f"a{i}", _embed(f"a{i}"), "doc-a", "a", "t")
    assert manager.remove_document("agent-x", "doc-a", keep_chunk_ids={"a1"}) == 2
    assert set(manager.get_store("agent-x").chunk_metadata) == {"a1"}


def test_reupload_replaces_old_vectors(tmp_path, daos):
    context_dao, _ = daos
    ok, doc_id = _upload(tmp_path, "alpha alpha alpha")
    assert ok
    ok, doc_id_2 = _upload(tmp_path, "beta beta beta")
    assert ok
    assert doc_id_2 == doc_id

    collection = context_dao.collection
    rows = list(collection.find({"agent_id": "agent-x"}))
    assert [r["text"] for r in rows] == ["beta beta beta"]

    store = FAISSVectorStoreManager().get_store("agent-x")
    assert store.size() == len(rows)
    assert {m["text"] for m in store.chunk_metadata.values()} == {"beta beta beta"}

    document_map: dict = {}
    vector_search(
        "agent-x",
        _embed("alpha alpha alpha"),
        [doc_id],
        collection,
        10,
        5,
        document_map,
    )
    assert all("alpha" not in d["text"] for d in document_map.values())


def test_failed_reupload_keeps_old_version(tmp_path, daos, monkeypatch):
    context_dao, _ = daos
    ok, _ = _upload(tmp_path, "alpha alpha alpha")
    assert ok

    def failing_embed(*_a, **_k):
        raise EmbeddingError("embedding service down")

    monkeypatch.setattr(context_upload, "compute_embedding", failing_embed)
    with pytest.raises(EmbeddingError):
        _upload(tmp_path, "beta beta beta")

    rows = list(context_dao.collection.find({"agent_id": "agent-x"}))
    assert [r["text"] for r in rows] == ["alpha alpha alpha"]
    store = FAISSVectorStoreManager().get_store("agent-x")
    assert {m["text"] for m in store.chunk_metadata.values()} == {"alpha alpha alpha"}


def test_reupload_with_faiss_failure_rolls_back_new_rows(tmp_path, daos, monkeypatch):
    context_dao, _ = daos
    ok, _ = _upload(tmp_path, "alpha alpha alpha")
    assert ok

    # Wrong dimension: the Mongo row is written, then the FAISS add raises.
    monkeypatch.setattr(context_upload, "compute_embedding", lambda *a, **k: [1.0] * 4)
    ok, _ = _upload(tmp_path, "beta beta beta")
    assert not ok

    rows = list(context_dao.collection.find({"agent_id": "agent-x"}))
    assert [r["text"] for r in rows] == ["alpha alpha alpha"]
    store = FAISSVectorStoreManager().get_store("agent-x")
    assert store.size() == 1


def test_document_delete_removes_vectors(tmp_path, daos):
    context_dao, document_dao = daos
    ok, doc_id = _upload(tmp_path, "alpha alpha alpha")
    ok_other, other_id = _upload(tmp_path, "gamma gamma", name="other.txt")
    assert ok and ok_other

    assert document_dao.delete(doc_id)

    assert context_dao.collection.count_documents({"document_id": doc_id}) == 0
    FAISSVectorStoreManager._stores.clear()
    store = FAISSVectorStore("agent-x")
    assert store.size() == 1
    assert {m["document_id"] for m in store.chunk_metadata.values()} == {other_id}


def _write_legacy_index(agent_id: str, vectors: dict[str, list[float]]) -> None:
    """An IndexFlatL2 file and the old metadata layout (chunk_id -> metadata)."""
    index = faiss.IndexFlatL2(DIM)
    matrix = np.array(list(vectors.values()), dtype=np.float32)
    faiss.normalize_L2(matrix)
    index.add(matrix)
    faiss.write_index(index, str(faiss_vector_store.get_index_path(agent_id)))
    metadata = {
        chunk_id: {
            "document_id": "doc-1",
            "document_name": "doc.txt",
            "text": f"text {chunk_id}",
            "chunk_index": i,
            "total_chunks": len(vectors),
        }
        for i, chunk_id in enumerate(vectors)
    }
    with open(faiss_vector_store.get_metadata_path(agent_id), "wb") as f:
        np.save(f, metadata)


def test_legacy_l2_index_loads_and_scores_correctly(caplog):
    u, v = _pair_with_cosine(0.8)
    _write_legacy_index("agent-x", {"old-0": v, "old-1": _embed("other")})

    with caplog.at_level(logging.WARNING):
        scores = FAISSVectorStoreManager().search("agent-x", u, ["doc-1"], top_k=2)

    assert scores["old-0"] == pytest.approx(0.8, abs=1e-5)
    assert "rebuild" in caplog.text.lower()


def test_add_to_legacy_index_converts_it(caplog):
    u, v = _pair_with_cosine(0.8)
    _write_legacy_index("agent-x", {"old-0": v, "old-1": _embed("other")})
    manager = FAISSVectorStoreManager()

    manager.add_vector("agent-x", "new-0", _embed("new"), "doc-2", "n", "t")

    FAISSVectorStoreManager._stores.clear()
    store = FAISSVectorStore("agent-x")
    assert isinstance(store.index, faiss.IndexIDMap2)
    assert set(store.chunk_metadata) == {"old-0", "old-1", "new-0"}
    scores = manager.search("agent-x", u, ["doc-1"], top_k=1)
    assert scores["old-0"] == pytest.approx(0.8, abs=1e-5)
    assert manager.remove_document("agent-x", "doc-1") == 2
    assert set(manager.get_store("agent-x").chunk_metadata) == {"new-0"}


def test_rebuild_converts_legacy_index_to_new_format():
    _write_legacy_index("agent-x", {"old-0": _embed("x")})
    collection = get_context_dao().collection
    collection.insert_one(
        {
            "agent_id": "agent-x",
            "document_id": "doc-1",
            "chunk_id": "doc-1:0:abc",
            "chunk_index": 0,
            "text": "t",
            "embedding": _embed("y"),
        }
    )

    assert FAISSVectorStoreManager().rebuild_index("agent-x", collection) == 1

    FAISSVectorStoreManager._stores.clear()
    store = FAISSVectorStore("agent-x")
    assert isinstance(store.index, faiss.IndexIDMap2)
    assert set(store.chunk_metadata) == {"doc-1:0:abc"}


def test_id_map_out_of_step_with_index_is_corrupt():
    manager = FAISSVectorStoreManager()
    manager.add_vector("agent-x", "c0", _embed("a"), "doc-1", "n", "t")
    manager.add_vector("agent-x", "c1", _embed("b"), "doc-1", "n", "t")
    meta_path = faiss_vector_store.get_metadata_path("agent-x")
    metadata = np.load(str(meta_path), allow_pickle=True).item()
    # Same entry count, but an id the index does not hold.
    metadata["id_to_chunk"] = {99: "c0", 1: "c1"}
    with open(meta_path, "wb") as f:
        np.save(f, metadata)
    FAISSVectorStoreManager._stores.clear()

    with pytest.raises(CorruptIndexError):
        FAISSVectorStore("agent-x")
