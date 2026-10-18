import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src import context_upload
from src.rag_service.context import Context
from src.rag_service.dao.context import faiss_vector_store
from src.rag_service.dao.context.faiss_vector_store import (
    CorruptIndexError,
    FAISSVectorStore,
    FAISSVectorStoreManager,
    IndexDimensionMismatchError,
)
from src.rag_service.dao.context.hybrid_search import vector_search
from src.rag_service.dao.factory import get_context_dao


REPO_ROOT = Path(__file__).resolve().parents[1]


def _vec(seed: int, dim: int) -> list[float]:
    v = np.random.default_rng(seed).standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    path = tmp_path / "faiss"
    monkeypatch.setattr(faiss_vector_store, "FAISS_INDEX_DIR", path)
    monkeypatch.setattr(FAISSVectorStoreManager, "_stores", {})
    return path


def _clear_cache():
    FAISSVectorStoreManager._stores.clear()


def _add(manager, agent_id, n, dim):
    for i in range(n):
        manager.add_vector(
            agent_id,
            f"c{i}",
            _vec(i, dim),
            "doc-1",
            "doc.txt",
            f"text {i}",
            i,
            n,
            embedding_dim=dim,
        )


def test_cold_start_search_adopts_file_dimension_and_keeps_file(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 3, 16)
    index_file = faiss_vector_store.get_index_path("agent-x")
    before = index_file.read_bytes()

    _clear_cache()
    # No dimension given: it must come from the file, not a 384 default.
    results = manager.search("agent-x", _vec(0, 16), ["doc-1"], top_k=3)

    assert "c0" in results
    assert index_file.read_bytes() == before
    assert manager.get_store("agent-x").embedding_dim == 16


def test_store_without_requested_dim_adopts_index_d(index_dir):
    FAISSVectorStoreManager().add_vector(
        "agent-x", "c0", _vec(0, 8), "doc-1", "n", "t", embedding_dim=8
    )
    store = FAISSVectorStore("agent-x")
    assert store.embedding_dim == 8
    assert store.size() == 1


def test_search_with_other_dimension_raises_and_keeps_file(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 2, 16)
    index_file = faiss_vector_store.get_index_path("agent-x")
    meta_file = faiss_vector_store.get_metadata_path("agent-x")
    before = (index_file.read_bytes(), meta_file.read_bytes())

    _clear_cache()
    with pytest.raises(IndexDimensionMismatchError) as exc:
        manager.search("agent-x", _vec(0, 8), ["doc-1"], embedding_dim=8)
    message = str(exc.value)
    assert "agent-x" in message and "16" in message and "8" in message

    assert (index_file.read_bytes(), meta_file.read_bytes()) == before
    # The file is still usable afterwards.
    _clear_cache()
    assert manager.search("agent-x", _vec(1, 16), ["doc-1"], embedding_dim=16)


def test_cached_store_with_other_dimension_raises(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 1, 16)
    with pytest.raises(IndexDimensionMismatchError):
        manager.get_store("agent-x", embedding_dim=8)
    with pytest.raises(IndexDimensionMismatchError):
        manager.search("agent-x", _vec(0, 8), ["doc-1"])


def test_add_with_other_dimension_raises(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 1, 16)
    _clear_cache()
    with pytest.raises(IndexDimensionMismatchError):
        manager.add_vector(
            "agent-x", "c9", _vec(9, 8), "doc-1", "n", "t", embedding_dim=8
        )
    assert FAISSVectorStore("agent-x").size() == 1


def test_empty_store_does_not_fix_dimension_on_search(index_dir):
    manager = FAISSVectorStoreManager()
    assert manager.search("agent-new", _vec(0, 8), [], embedding_dim=8) == {}
    # First upload decides the dimension, even after a differently sized search.
    _add(manager, "agent-new", 1, 16)
    assert manager.get_store("agent-new").embedding_dim == 16


def test_corrupt_index_is_renamed_and_raises(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 2, 8)
    index_file = faiss_vector_store.get_index_path("agent-x")
    index_file.write_bytes(b"not a faiss index")
    _clear_cache()

    with pytest.raises(CorruptIndexError):
        manager.search("agent-x", _vec(0, 8), ["doc-1"])

    assert not index_file.exists()
    corrupt = list(index_dir.glob("agent_agent-x.index.corrupt-*"))
    assert len(corrupt) == 1
    assert corrupt[0].read_bytes() == b"not a faiss index"
    assert list(index_dir.glob("agent_agent-x_metadata.npy.corrupt-*"))


def test_metadata_out_of_sync_with_index_is_treated_as_corrupt(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 2, 8)
    faiss_vector_store.get_metadata_path("agent-x").unlink()
    _clear_cache()

    with pytest.raises(CorruptIndexError):
        FAISSVectorStore("agent-x")
    assert list(index_dir.glob("agent_agent-x.index.corrupt-*"))


def test_save_is_atomic_and_leaves_no_temp_files(index_dir):
    _add(FAISSVectorStoreManager(), "agent-x", 2, 8)
    names = sorted(p.name for p in index_dir.iterdir())
    assert names == ["agent_agent-x.index", "agent_agent-x_metadata.npy"]


def test_vector_search_passes_query_dimension(index_dir):
    collection = get_context_dao().collection
    collection.insert_one({"chunk_id": "c0", "agent_id": "agent-x", "text": "t"})
    _add(FAISSVectorStoreManager(), "agent-x", 1, 16)
    _clear_cache()

    scores = vector_search("agent-x", _vec(0, 16), ["doc-1"], collection, 10, 5, {})
    assert "c0" in scores

    _clear_cache()
    with pytest.raises(IndexDimensionMismatchError):
        vector_search("agent-x", _vec(0, 8), ["doc-1"], collection, 10, 5, {})


def test_insert_context_reraises_faiss_failure(index_dir):
    dao = get_context_dao()
    _add(FAISSVectorStoreManager(), "agent-x", 1, 16)
    with pytest.raises(IndexDimensionMismatchError):
        dao.insert_context(
            "doc-2",
            "agent-x",
            _vec(5, 8),
            Context(text="t", document_name="n", chunk_id="doc-2:0:x"),
        )


def test_upload_reports_failure_when_faiss_add_fails(index_dir, tmp_path, monkeypatch):
    dao = get_context_dao()
    monkeypatch.setattr(context_upload, "get_context_dao", lambda: dao)
    monkeypatch.setattr(context_upload, "compute_embedding", lambda *a, **k: _vec(1, 8))
    _add(FAISSVectorStoreManager(), "agent-x", 1, 16)

    path = tmp_path / "notes.txt"
    path.write_text("Some content to upload.")
    ok, _ = context_upload.process_file_and_store(str(path), "agent-x", "stub:model")
    assert not ok


def _rows(agent_id, n, dim, doc="doc-1", seed0=0):
    return [
        {
            "agent_id": agent_id,
            "document_id": doc,
            "document_name": f"{doc}.txt",
            "chunk_id": f"{doc}:{i}:{agent_id}",
            "chunk_index": i,
            "total_chunks": n,
            "text": f"{agent_id} text {i}",
            "embedding": _vec(seed0 + i, dim),
        }
        for i in range(n)
    ]


def test_rebuild_index_from_mongo_rows(index_dir):
    collection = get_context_dao().collection
    collection.insert_many(_rows("agent-x", 3, 16) + _rows("agent-y", 2, 8))
    # An existing broken file must be replaced by the rebuild.
    index_dir.mkdir(parents=True)
    faiss_vector_store.get_index_path("agent-x").write_bytes(b"garbage")

    manager = FAISSVectorStoreManager()
    count = manager.rebuild_index("agent-x", collection)

    assert count == 3
    _clear_cache()
    store = manager.get_store("agent-x")
    assert store.embedding_dim == 16
    assert store.size() == 3
    results = manager.search("agent-x", _vec(1, 16), ["doc-1"], top_k=1)
    assert list(results) == ["doc-1:1:agent-x"]
    assert not faiss_vector_store.get_index_path("agent-y").exists()


def test_rebuild_with_no_rows_leaves_existing_files(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 1, 8)
    before = faiss_vector_store.get_index_path("agent-x").read_bytes()

    assert manager.rebuild_index("agent-x", get_context_dao().collection) == 0
    assert faiss_vector_store.get_index_path("agent-x").read_bytes() == before


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "rebuild_faiss_index", REPO_ROOT / "scripts" / "rebuild_faiss_index.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuild_cli_single_agent_and_all(index_dir, monkeypatch):
    dao = get_context_dao()
    dao.collection.insert_many(_rows("agent-x", 2, 16) + _rows("agent-y", 3, 8))
    script = _load_script()
    monkeypatch.setattr(script, "get_context_dao", lambda: dao)

    assert script.main(["--agent-id", "agent-x"]) == 0
    assert FAISSVectorStore("agent-x").size() == 2
    assert not faiss_vector_store.get_index_path("agent-y").exists()

    assert script.main(["--all"]) == 0
    assert FAISSVectorStore("agent-x").size() == 2
    assert FAISSVectorStore("agent-y").size() == 3


def test_rebuild_cli_requires_agent_or_all():
    script = _load_script()
    with pytest.raises(SystemExit):
        script.main([])
    with pytest.raises(SystemExit):
        script.main(["--agent-id", "a", "--all"])


def test_rebuild_skips_duplicate_chunk_ids(index_dir):
    collection = get_context_dao().collection
    rows = _rows("agent-x", 2, 8)
    dup = dict(rows[0], embedding=_vec(99, 8))
    collection.insert_many([*rows, dup])

    assert FAISSVectorStoreManager().rebuild_index("agent-x", collection) == 2
    _clear_cache()
    assert FAISSVectorStore("agent-x").size() == 2


def _rebuild_in_other_process(agent_id, collection):
    # The manager is a singleton with a class-level cache; swapping the cache
    # simulates the CLI running in its own process with its own cache.
    server_cache = FAISSVectorStoreManager._stores
    FAISSVectorStoreManager._stores = {}
    try:
        return FAISSVectorStoreManager().rebuild_index(agent_id, collection)
    finally:
        FAISSVectorStoreManager._stores = server_cache


def test_cached_store_picks_up_rebuild_from_other_process_on_search(index_dir):
    collection = get_context_dao().collection
    manager = FAISSVectorStoreManager()
    # Server cached an empty store (e.g. after a CorruptIndexError).
    assert manager.search("agent-x", _vec(0, 16), ["doc-1"]) == {}
    collection.insert_many(_rows("agent-x", 3, 16))

    assert _rebuild_in_other_process("agent-x", collection) == 3

    results = manager.search("agent-x", _vec(1, 16), ["doc-1"], top_k=1)
    assert list(results) == ["doc-1:1:agent-x"]


def test_add_on_stale_cache_keeps_rebuilt_vectors(index_dir):
    collection = get_context_dao().collection
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 1, 16)  # stale cached store: 1 vector, "c0"
    collection.insert_many(_rows("agent-x", 3, 16))

    assert _rebuild_in_other_process("agent-x", collection) == 3

    manager.add_vector(
        "agent-x", "new-chunk", _vec(50, 16), "doc-1", "n", "t", embedding_dim=16
    )
    assert manager.get_store("agent-x").size() == 4
    _clear_cache()
    on_disk = FAISSVectorStore("agent-x")
    assert on_disk.size() == 4
    assert set(on_disk.chunk_metadata) == {
        "doc-1:0:agent-x",
        "doc-1:1:agent-x",
        "doc-1:2:agent-x",
        "new-chunk",
    }


def test_add_rejects_duplicate_chunk_id(index_dir):
    manager = FAISSVectorStoreManager()
    _add(manager, "agent-x", 1, 8)
    with pytest.raises(faiss_vector_store.FAISSIndexError):
        manager.add_vector("agent-x", "c0", _vec(7, 8), "doc-1", "n", "t")
    _clear_cache()
    assert FAISSVectorStore("agent-x").size() == 1
