"""Local FAISS-based vector store for self-contained vector search.

This module provides vector storage and retrieval using FAISS (Facebook AI Similarity Search)
with cosine similarity. It replaces MongoDB Atlas Vector Search for self-contained deployments.

Key Features:
- Local vector indexing with FAISS (no external dependencies)
- Cosine similarity search: IndexIDMap2 over IndexFlatIP with L2-normalized
  vectors, so the inner product is the cosine similarity
- Stable int64 id per vector (``id_to_chunk`` in the metadata file), so vectors
  can be removed (documents replaced or deleted) without shifting the others
- Per-agent vector indices (isolated namespaces)
- Persistence to disk (atomic writes; files are never deleted on load errors)
- The index dimension comes from the file, or from the first vector added
- Rebuild from the MongoDB context rows (see scripts/rebuild_faiss_index.py)

Thread safety: every store object of an agent shares one re-entrant lock
(see ``_agent_lock``), held across refresh, add, remove, search and save.
Without it a chat thread could refresh between this process's own index and
metadata writes, take the half-finished save for an outside change, and
quarantine the index. Changes by other processes are still detected by
``refresh()``.

Legacy indexes (IndexFlatL2 + a ``chunk_id -> metadata`` dict, where the
position of a vector was its dict order) still load and search. The first add or
remove converts them in memory to the id-mapped format before saving; a rebuild
from MongoDB is recommended because they may hold vectors of replaced documents.
"""

import logging
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


logger = logging.getLogger(__name__)

# What reading a broken or half-written index/metadata file can raise
# (faiss raises RuntimeError).
_READ_ERRORS = (
    RuntimeError,
    OSError,
    ValueError,
    TypeError,
    EOFError,
    pickle.PickleError,
)

# Layout version of the metadata file written by this module. Files without
# "format_version" are legacy (IndexFlatL2, positions in dict order).
METADATA_FORMAT_VERSION = 2

# Directory for storing FAISS indices
FAISS_INDEX_DIR = Path("/data/faiss_indices")


class FAISSIndexError(RuntimeError):
    """Base class for FAISS index problems that must not be silently ignored."""


class IndexDimensionMismatchError(FAISSIndexError, ValueError):
    """Raised when a vector's dimension differs from the agent's index.

    The index file is never deleted on a mismatch. If the agent's embedding
    model really changed, rebuild the index with scripts/rebuild_faiss_index.py.
    """

    def __init__(self, agent_id: str, index_dim: int, requested_dim: int):
        self.agent_id = agent_id
        self.index_dim = index_dim
        self.requested_dim = requested_dim
        super().__init__(
            f"FAISS index for agent {agent_id} has dimension {index_dim}, "
            f"but dimension {requested_dim} was requested. The index was left "
            f"untouched; check the agent's embedding model or rebuild the index."
        )


class CorruptIndexError(FAISSIndexError):
    """Raised when an index file cannot be loaded; the files are set aside."""


# One lock per agent, shared by every FAISSVectorStore object of that agent
# (a rebuild creates a second object for the same files).
_agent_locks: dict[str, threading.RLock] = {}
_agent_locks_guard = threading.Lock()


def _agent_lock(agent_id: str) -> threading.RLock:
    with _agent_locks_guard:
        lock = _agent_locks.get(agent_id)
        if lock is None:
            lock = _agent_locks[agent_id] = threading.RLock()
        return lock


def ensure_index_dir() -> Path:
    """Ensure the FAISS index directory exists."""
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return FAISS_INDEX_DIR


def get_index_path(agent_id: str) -> Path:
    """Get the file path for an agent's FAISS index.

    Args:
        agent_id: The agent ID

    Returns:
        Path to the FAISS index file
    """
    ensure_index_dir()
    return FAISS_INDEX_DIR / f"agent_{agent_id}.index"


def get_metadata_path(agent_id: str) -> Path:
    """Get the file path for an agent's chunk metadata.

    Args:
        agent_id: The agent ID

    Returns:
        Path to the metadata file
    """
    ensure_index_dir()
    return FAISS_INDEX_DIR / f"agent_{agent_id}_metadata.npy"


def _save_metadata(path: Path, metadata: dict) -> None:
    # Through a file handle so numpy does not append another ".npy".
    with open(path, "wb") as f:
        np.save(f, metadata)


def _file_signature(path: Path) -> tuple[int, int, int] | None:
    """Identity of a file on disk (None when missing), to spot outside changes."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _atomic_write(target: Path, write) -> None:
    """Call ``write(tmp_path)`` and atomically move the result onto ``target``."""
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class FAISSVectorStore:
    """Local vector store using FAISS for cosine similarity search.

    This class manages vector indices for individual agents, supporting
    efficient similarity search without external dependencies.
    """

    def __init__(
        self,
        agent_id: str,
        embedding_dim: int | None = None,
        load_existing: bool = True,
    ):
        """Initialize FAISS vector store for an agent.

        Args:
            agent_id: The agent ID
            embedding_dim: Expected dimension, if known. When an index file
                exists its dimension wins, and a different expected dimension
                raises IndexDimensionMismatchError. Without a file the
                dimension is fixed by the first vector added.
            load_existing: False starts empty without reading the file (used
                by rebuilds); the file is only replaced on the next save.

        Raises:
            IndexDimensionMismatchError: The file has another dimension.
            CorruptIndexError: The file could not be loaded (it was renamed).
        """
        self.agent_id = agent_id
        self._lock = _agent_lock(agent_id)
        self.index: faiss.Index | None = None
        # chunk_id -> {int_id, document_id, document_name, text, ...}
        self.chunk_metadata: dict = {}
        # FAISS vector id -> chunk_id. Ids are never reused.
        self._id_to_chunk: dict[int, str] = {}
        self._next_id = 0
        # Old IndexFlatL2 file: ids are positions, scores need 1 - d/2.
        self._legacy = False
        # Identity of the files this copy matches (from the last load or save).
        # Another process (the rebuild CLI, another worker) may replace them;
        # refresh() then reloads instead of serving or overwriting stale data.
        # A store started with load_existing=False ignores the disk until it
        # saves for the first time.
        self._sync_with_disk = load_existing
        self._disk_signature: tuple | None = None
        if load_existing:
            with self._lock:
                self._load_index(embedding_dim)

    @property
    def embedding_dim(self) -> int | None:
        """Dimension of the index, or None while the store is still empty."""
        return self.index.d if self.index is not None else None

    def check_dimension(self, requested_dim: int | None) -> None:
        """Raise if a requested dimension differs from this store's index."""
        if (
            requested_dim is not None
            and self.embedding_dim is not None
            and requested_dim != self.embedding_dim
        ):
            raise IndexDimensionMismatchError(
                self.agent_id, self.embedding_dim, requested_dim
            )

    def _current_disk_signature(self) -> tuple:
        return (
            _file_signature(get_index_path(self.agent_id)),
            _file_signature(get_metadata_path(self.agent_id)),
        )

    def refresh(self) -> None:
        """Reload from disk if the files changed since this copy was loaded or saved.

        Costs two stat calls. If the files disappeared, the store becomes empty
        to match the disk. Runs under the agent lock, and saves by this process
        update the signature under the same lock, so they are never mistaken
        for an outside change.
        """
        with self._lock:
            if not self._sync_with_disk:
                return
            if self._current_disk_signature() == self._disk_signature:
                return
            logger.info(
                f"FAISS index files for agent {self.agent_id} changed on disk; "
                f"reloading"
            )
            self._reset()
            self._load_index(None)

    def _reset(self) -> None:
        self.index = None
        self.chunk_metadata = {}
        self._id_to_chunk = {}
        self._next_id = 0
        self._legacy = False

    def _load_index(self, requested_dim: int | None) -> None:
        """Load the agent's index from disk, if there is one.

        Never deletes or overwrites files: a dimension mismatch raises, and an
        unreadable index is renamed to ``*.corrupt-<timestamp>`` before raising.
        """
        index_path = get_index_path(self.agent_id)
        metadata_path = get_metadata_path(self.agent_id)

        signature = self._current_disk_signature()
        if not index_path.exists():
            self._disk_signature = signature
            logger.info(f"No FAISS index on disk yet for agent {self.agent_id}")
            return

        try:
            try:
                loaded = self._read_files(index_path, metadata_path)
            except _READ_ERRORS:
                # Another process may be between replacing the index and the
                # metadata file; give it a moment before calling it corrupt.
                time.sleep(0.2)
                signature = self._current_disk_signature()
                loaded = self._read_files(index_path, metadata_path)
        except Exception as e:
            suffix = f".corrupt-{time.strftime('%Y%m%d-%H%M%S')}"
            for path in (index_path, metadata_path):
                if path.exists():
                    path.replace(path.with_name(path.name + suffix))
            logger.error(
                f"Could not load FAISS index for agent {self.agent_id}: {e}. "
                f"Files renamed with suffix {suffix}; rebuild the index from MongoDB."
            )
            raise CorruptIndexError(
                f"FAISS index for agent {self.agent_id} could not be loaded ({e}); "
                f"files were renamed with suffix {suffix}. "
                f"Run scripts/rebuild_faiss_index.py --agent-id {self.agent_id}."
            ) from e

        index, chunks, id_to_chunk, next_id, legacy = loaded
        self.index = index
        self.chunk_metadata = chunks
        self._id_to_chunk = id_to_chunk
        self._next_id = next_id
        self._legacy = legacy
        self._disk_signature = signature
        logger.info(
            f"Loaded FAISS index for agent {self.agent_id} with {index.ntotal} "
            f"vectors of dimension {index.d}"
        )
        if legacy:
            logger.warning(
                f"FAISS index for agent {self.agent_id} uses the legacy L2 format; "
                f"it may still hold vectors of replaced documents. Rebuild it with "
                f"scripts/rebuild_faiss_index.py --agent-id {self.agent_id}"
            )
        self.check_dimension(requested_dim)

    @staticmethod
    def _read_files(index_path: Path, metadata_path: Path) -> tuple:
        """Read and cross-check the index and metadata files.

        Returns:
            (index, chunk_metadata, id_to_chunk, next_id, is_legacy)
        """
        index = faiss.read_index(str(index_path))
        metadata = (
            np.load(str(metadata_path), allow_pickle=True).item()
            if metadata_path.exists()
            else {}
        )
        if not isinstance(metadata, dict):
            raise TypeError("metadata file does not hold a dict")

        if "format_version" not in metadata:
            # Legacy: chunk_id -> metadata, vector position = dict order.
            if isinstance(index, faiss.IndexIDMap2):
                raise ValueError("id-mapped index with legacy metadata")
            if len(metadata) != index.ntotal:
                raise ValueError(
                    f"metadata has {len(metadata)} entries but the index has "
                    f"{index.ntotal} vectors"
                )
            id_to_chunk = dict(enumerate(metadata))
            return index, metadata, id_to_chunk, len(id_to_chunk), True

        if metadata["format_version"] != METADATA_FORMAT_VERSION:
            raise ValueError(
                f"unknown metadata format_version {metadata['format_version']}"
            )
        if not isinstance(index, faiss.IndexIDMap2):
            raise TypeError("metadata expects an id-mapped index")
        chunks = metadata["chunks"]
        id_to_chunk = {int(k): v for k, v in metadata["id_to_chunk"].items()}
        index_ids = {int(i) for i in faiss.vector_to_array(index.id_map)}
        # Every vector id must map to a chunk, and every chunk to a vector.
        if (
            len(id_to_chunk) != index.ntotal
            or set(id_to_chunk) != index_ids
            or set(id_to_chunk.values()) != set(chunks)
        ):
            raise ValueError(
                f"metadata ({len(id_to_chunk)} ids, {len(chunks)} chunks) does not "
                f"match the index ({index.ntotal} vectors)"
            )
        next_id = int(metadata["next_id"])
        if id_to_chunk and next_id <= max(id_to_chunk):
            raise ValueError("metadata next_id would reuse an existing id")
        return index, chunks, id_to_chunk, next_id, False

    def _create_new_index(self, embedding_dim: int) -> None:
        """Create a new FAISS index with cosine similarity."""
        # Vectors are L2-normalized, so inner product == cosine similarity.
        # The id map gives each vector a stable id for removal.
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(embedding_dim))
        self._id_to_chunk = {}
        self._next_id = 0
        self._legacy = False
        logger.info(
            f"Created new {embedding_dim}-dim FAISS index for agent {self.agent_id}"
        )

    def _convert_legacy(self) -> None:
        """Convert a loaded legacy IndexFlatL2 to the id-mapped format in memory.

        The stored vectors are already normalized, so they are copied as is;
        their positions become their ids. Nothing is written until the next save.
        """
        old = self.index
        n = old.ntotal
        new = faiss.IndexIDMap2(faiss.IndexFlatIP(old.d))
        if n:
            vectors = old.reconstruct_n(0, n)
            new.add_with_ids(vectors, np.arange(n, dtype=np.int64))
        for int_id, chunk_id in self._id_to_chunk.items():
            self.chunk_metadata[chunk_id]["int_id"] = int_id
        self.index = new
        self._next_id = n
        self._legacy = False
        logger.warning(
            f"Converted legacy FAISS index for agent {self.agent_id} "
            f"({n} vectors) to the id-mapped format. It may still hold vectors "
            f"of replaced documents; rebuild it with "
            f"scripts/rebuild_faiss_index.py --agent-id {self.agent_id}"
        )

    def add_vectors(
        self,
        chunk_id: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
        text: str,
        chunk_index: int | None = None,
        total_chunks: int | None = None,
        persist: bool = True,
    ) -> None:
        """Add a vector to the index.

        Args:
            chunk_id: Unique chunk identifier
            embedding: The embedding vector (will be normalized)
            document_id: Document this chunk belongs to
            document_name: Name of the document
            text: The chunk text content
            chunk_index: Index of chunk within document
            total_chunks: Total number of chunks in document
            persist: Save to disk after adding. False for bulk rebuilds,
                which save once at the end.

        Raises:
            IndexDimensionMismatchError: The embedding has another dimension.
            FAISSIndexError: The chunk id is already in the index.
        """
        with self._lock:
            self._add_vector_locked(
                chunk_id,
                embedding,
                document_id,
                document_name,
                text,
                chunk_index,
                total_chunks,
                persist,
            )

    def _add_vector_locked(
        self,
        chunk_id: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
        text: str,
        chunk_index: int | None,
        total_chunks: int | None,
        persist: bool,
    ) -> None:
        self.refresh()
        if chunk_id in self.chunk_metadata:
            # Metadata is keyed by chunk id; a second vector under the same id
            # would desync index and metadata (and every later result).
            raise FAISSIndexError(
                f"Chunk {chunk_id} is already in the FAISS index for agent "
                f"{self.agent_id}"
            )
        snapshot = self._snapshot() if persist else None
        if self.index is None:
            self._create_new_index(len(embedding))
        elif self._legacy:
            self._convert_legacy()

        # Validate embedding dimensions
        if len(embedding) != self.embedding_dim:
            logger.error(
                f"Embedding dimension mismatch for chunk {chunk_id}: "
                f"expected {self.embedding_dim}, got {len(embedding)}"
            )
            raise IndexDimensionMismatchError(
                self.agent_id, self.embedding_dim, len(embedding)
            )

        # Normalize vector for cosine similarity
        vector = np.array([embedding], dtype=np.float32)
        size_before = self.index.ntotal
        faiss.normalize_L2(vector)

        # Add to FAISS index under a fresh id
        int_id = self._next_id
        self.index.add_with_ids(vector, np.array([int_id], dtype=np.int64))
        self._next_id += 1
        size_after = self.index.ntotal

        # Store metadata
        self._id_to_chunk[int_id] = chunk_id
        self.chunk_metadata[chunk_id] = {
            "int_id": int_id,
            "document_id": document_id,
            "document_name": document_name,
            "text": text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }

        # Save index and metadata; on failure undo the add so memory never
        # runs ahead of the disk.
        if persist:
            self._save_or_restore(snapshot)

        logger.info(
            f"Added chunk {chunk_id} to FAISS index for agent {self.agent_id}. "
            f"Index size: {size_before} -> {size_after}. "
            f"Document: {document_name} (doc_id: {document_id})"
        )

    def search(
        self,
        query_embedding: list[float],
        available_documents: list[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> dict[str, float]:
        """Search for similar vectors using cosine similarity.

        Args:
            query_embedding: Query embedding vector (will be normalized)
            available_documents: List of document IDs to filter by
            top_k: Number of results to return
            similarity_threshold: Minimum cosine similarity; the default 0.0
                drops vectors pointing away from the query

        Returns:
            Dictionary mapping chunk_id -> cosine similarity in [-1, 1]

        Raises:
            IndexDimensionMismatchError: The query has another dimension.
        """
        with self._lock:
            return self._search_locked(
                query_embedding, available_documents, top_k, similarity_threshold
            )

    def _search_locked(
        self,
        query_embedding: list[float],
        available_documents: list[str],
        top_k: int,
        similarity_threshold: float,
    ) -> dict[str, float]:
        self.refresh()
        if not self.index or self.index.ntotal == 0:
            logger.warning(f"No vectors in index for agent {self.agent_id}")
            return {}

        # Outside the try below: a mismatch must reach the caller instead of
        # looking like "no results".
        self.check_dimension(len(query_embedding))

        try:
            # Log query details
            logger.debug(
                f"FAISS search for agent {self.agent_id}: query_embedding type={type(query_embedding)}, available_documents={available_documents}"
            )

            # Handle query_embedding - convert to properly shaped numpy array
            if isinstance(query_embedding, np.ndarray):
                # If already numpy array
                if query_embedding.ndim == 1:
                    query_vector = query_embedding.reshape(1, -1).astype(np.float32)
                else:
                    query_vector = query_embedding.astype(np.float32)
            elif isinstance(query_embedding, (list, tuple)):
                # If list or tuple, convert to 2D array
                query_vector = np.array([query_embedding], dtype=np.float32)
            else:
                logger.error(f"Invalid query_embedding type: {type(query_embedding)}")
                return {}

            # Ensure it's 2D (batch of vectors)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            elif query_vector.ndim != 2:
                logger.error(f"Invalid query_vector shape: {query_vector.shape}")
                return {}

            # Normalize query vector for cosine similarity
            faiss.normalize_L2(query_vector)

            # Convert available_documents to set for faster lookup
            available_docs_set = (
                set(available_documents) if available_documents else set()
            )

            # Search - get more candidates to account for filtering
            search_k = min(top_k * 5, self.index.ntotal)
            distances, indices = self.index.search(query_vector, search_k)

            logger.debug(
                f"FAISS raw search for agent {self.agent_id}: index_size={self.index.ntotal}, search_k={search_k}, raw_results={len(indices[0])}"
            )

            logger.debug(
                f"FAISS metadata for agent {self.agent_id}: chunk_ids_count={len(self._id_to_chunk)}, available_docs={available_docs_set}"
            )

            results = {}
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                chunk_id = self._id_to_chunk.get(int(idx))
                if chunk_id is None:
                    logger.debug(f"  Result {i}: unknown id {idx}")
                    continue

                metadata = self.chunk_metadata.get(chunk_id, {})

                # Filter by available documents
                doc_id = metadata.get("document_id")
                if self._legacy:
                    # IndexFlatL2 returns squared L2 distances; for unit
                    # vectors d = 2 - 2*cos.
                    cosine_similarity = 1.0 - float(dist) / 2.0
                else:
                    # Inner product of unit vectors is the cosine.
                    cosine_similarity = float(dist)
                cosine_similarity = min(1.0, max(-1.0, cosine_similarity))

                if not available_docs_set:
                    logger.debug(
                        f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f} (no filter)"
                    )
                elif doc_id not in available_docs_set:
                    logger.debug(
                        f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f} FILTERED OUT"
                    )
                    continue
                else:
                    logger.debug(
                        f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f}"
                    )

                # Apply threshold
                if cosine_similarity >= similarity_threshold:
                    results[chunk_id] = cosine_similarity
                else:
                    logger.debug(f"    Below threshold ({similarity_threshold})")

            # Sort by similarity and return top_k
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[
                :top_k
            ]
            logger.info(
                f"FAISS search for agent {self.agent_id}: found {len(sorted_results)} final results (after filtering)"
            )
            return dict(sorted_results)

        except Exception as e:
            logger.error(
                f"Error in FAISS search for agent {self.agent_id}: {e}", exc_info=True
            )
            return {}

    def _save_index(self) -> None:
        """Persist the index and metadata to disk atomically.

        Each file is written to a temp file in the same directory and moved
        into place with os.replace, so a crash never leaves a truncated file.
        Failures are raised: an unsaved vector would be silently lost.
        The agent lock is held throughout and the disk signature is updated
        under it, so refresh() never sees this save half-done.
        """
        with self._lock:
            self._save_index_locked()

    def _save_index_locked(self) -> None:
        if not self.index:
            return
        if self._legacy:
            # Never write the new metadata layout next to an L2 index.
            self._convert_legacy()

        index_path = get_index_path(self.agent_id)
        metadata_path = get_metadata_path(self.agent_id)

        try:
            _atomic_write(
                index_path, lambda tmp: faiss.write_index(self.index, str(tmp))
            )
            metadata = {
                "format_version": METADATA_FORMAT_VERSION,
                "next_id": self._next_id,
                "id_to_chunk": self._id_to_chunk,
                "chunks": self.chunk_metadata,
            }
            _atomic_write(metadata_path, lambda tmp: _save_metadata(tmp, metadata))
            self._disk_signature = self._current_disk_signature()
            self._sync_with_disk = True
            logger.debug(f"Saved FAISS index for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index for agent {self.agent_id}: {e}")
            raise FAISSIndexError(
                f"Failed to save FAISS index for agent {self.agent_id}: {e}"
            ) from e

    def remove_chunks(self, chunk_ids, persist: bool = True) -> int:
        """Remove the vectors of the given chunks; unknown ids are ignored.

        Returns:
            Number of vectors removed.

        Raises:
            FAISSIndexError: Saving the index failed (nothing is removed).
        """
        with self._lock:
            return self._remove_chunks_locked(chunk_ids, persist)

    def _remove_chunks_locked(self, chunk_ids, persist: bool) -> int:
        self.refresh()
        targets = [c for c in set(chunk_ids) if c in self.chunk_metadata]
        if not targets:
            return 0
        snapshot = self._snapshot() if persist else None
        if self._legacy:
            # Positions of an IndexFlatL2 shift on removal; ids do not.
            self._convert_legacy()

        ids = [self.chunk_metadata[c]["int_id"] for c in targets]
        removed = self.index.remove_ids(np.array(ids, dtype=np.int64))
        for int_id, chunk_id in zip(ids, targets, strict=True):
            del self._id_to_chunk[int_id]
            del self.chunk_metadata[chunk_id]
        if persist:
            self._save_or_restore(snapshot)
        logger.info(
            f"Removed {removed} vectors from FAISS index for agent {self.agent_id}. "
            f"Index size: {self.index.ntotal}"
        )
        return int(removed)

    def remove_document(
        self, document_id: str, keep_chunk_ids=None, persist: bool = True
    ) -> int:
        """Remove every vector of a document except ``keep_chunk_ids``.

        Returns:
            Number of vectors removed.
        """
        with self._lock:
            self.refresh()
            keep = set(keep_chunk_ids or ())
            targets = [
                chunk_id
                for chunk_id, meta in self.chunk_metadata.items()
                if meta.get("document_id") == document_id and chunk_id not in keep
            ]
            return self.remove_chunks(targets, persist=persist)

    def _snapshot(self) -> tuple:
        """Copy of the in-memory state, to undo a change whose save failed."""
        return (
            faiss.clone_index(self.index) if self.index is not None else None,
            dict(self.chunk_metadata),
            dict(self._id_to_chunk),
            self._next_id,
            self._legacy,
            self._disk_signature,
        )

    def _save_or_restore(self, snapshot: tuple) -> None:
        """Save; if that fails, put the snapshot back and re-raise.

        The failed save may already have replaced the index file but not the
        metadata, so the restored state is written again (best effort) to keep
        the two files consistent for the next load.
        """
        try:
            self._save_index_locked()
        except FAISSIndexError:
            (
                self.index,
                self.chunk_metadata,
                self._id_to_chunk,
                self._next_id,
                self._legacy,
                old_signature,
            ) = snapshot
            try:
                if self.index is not None:
                    self._save_index_locked()
                elif old_signature is not None and old_signature[0] is None:
                    # There was no index file before this change.
                    for path in (
                        get_index_path(self.agent_id),
                        get_metadata_path(self.agent_id),
                    ):
                        path.unlink(missing_ok=True)
                    self._disk_signature = self._current_disk_signature()
            except Exception as e:  # noqa: BLE001 - the original error is raised
                logger.error(
                    f"Could not restore FAISS files for agent {self.agent_id} "
                    f"after a failed save ({type(e).__name__}); the next load "
                    f"may need a rebuild"
                )
            raise

    def get_metadata(self, chunk_id: str) -> dict | None:
        """Get metadata for a chunk.

        Args:
            chunk_id: The chunk ID

        Returns:
            Metadata dictionary or None if not found
        """
        with self._lock:
            return self.chunk_metadata.get(chunk_id)

    def size(self) -> int:
        """Get the number of vectors in the index."""
        with self._lock:
            return self.index.ntotal if self.index else 0


class FAISSVectorStoreManager:
    """Manages FAISS vector stores for multiple agents.

    This singleton manager maintains per-agent vector stores in memory
    and handles persistence.
    """

    _instance: Optional["FAISSVectorStoreManager"] = None
    _stores: dict[str, FAISSVectorStore] = {}

    def __new__(cls) -> "FAISSVectorStoreManager":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_store(
        self, agent_id: str, embedding_dim: int | None = None
    ) -> FAISSVectorStore:
        """Get or create a vector store for an agent.

        Args:
            agent_id: The agent ID
            embedding_dim: Expected dimension, if known. Taken from the index
                file when one exists.

        Returns:
            FAISSVectorStore instance for the agent

        Raises:
            IndexDimensionMismatchError: The agent's index has another dimension.
            CorruptIndexError: The index file could not be loaded.
        """
        # Only the per-agent lock is taken (never a manager-wide one), so
        # there is no lock order to get wrong.
        with _agent_lock(agent_id):
            store = self._stores.get(agent_id)
            if store is None:
                store = FAISSVectorStore(agent_id, embedding_dim)
                self._stores[agent_id] = store
            else:
                store.refresh()
                store.check_dimension(embedding_dim)
            return store

    def add_vector(
        self,
        agent_id: str,
        chunk_id: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
        text: str,
        chunk_index: int | None = None,
        total_chunks: int | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        """Add a vector to an agent's store.

        Args:
            agent_id: The agent ID
            chunk_id: Unique chunk identifier
            embedding: The embedding vector
            document_id: Document ID
            document_name: Document name
            text: Chunk text
            chunk_index: Chunk index within document
            total_chunks: Total chunks in document
            embedding_dim: Dimension of embeddings (defaults to len(embedding))
        """
        logger.info(
            f"FAISSVectorStoreManager.add_vector: agent={agent_id}, chunk={chunk_id}, embedding_dim={embedding_dim} (actual={len(embedding)}), doc={document_name}"
        )

        store = self.get_store(agent_id, embedding_dim or len(embedding))
        store.add_vectors(
            chunk_id,
            embedding,
            document_id,
            document_name,
            text,
            chunk_index,
            total_chunks,
        )

    def search(
        self,
        agent_id: str,
        query_embedding: list[float],
        available_documents: list[str],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        embedding_dim: int | None = None,
    ) -> dict[str, float]:
        """Search in an agent's vector store.

        Args:
            agent_id: The agent ID
            query_embedding: Query embedding
            available_documents: Documents to filter by
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            embedding_dim: Dimension of embeddings (defaults to the query's)

        Returns:
            Dictionary mapping chunk_id -> similarity_score
        """
        store = self.get_store(agent_id, embedding_dim or len(query_embedding))
        return store.search(
            query_embedding,
            available_documents,
            top_k,
            similarity_threshold,
        )

    def remove_document(
        self, agent_id: str, document_id: str, keep_chunk_ids=None
    ) -> int:
        """Remove a document's vectors from an agent's store (except ``keep_chunk_ids``).

        Returns:
            Number of vectors removed (0 when the agent has no index).
        """
        return self.get_store(agent_id).remove_document(document_id, keep_chunk_ids)

    def remove_chunks(self, agent_id: str, chunk_ids) -> int:
        """Remove the vectors of specific chunks from an agent's store.

        Returns:
            Number of vectors removed.
        """
        return self.get_store(agent_id).remove_chunks(chunk_ids)

    def rebuild_index(self, agent_id: str, context_collection=None) -> int:
        """Rebuild an agent's FAISS index from its MongoDB context rows.

        The rows store each chunk's embedding, so MongoDB can always recreate
        the index. The new index is built in memory through the normal add
        path and only replaces the files on disk once every row was added.
        With no rows, existing files are left alone. The result is always the
        id-mapped inner-product format, so this also migrates legacy indexes
        and drops vectors whose Mongo rows are gone.

        Args:
            agent_id: The agent ID
            context_collection: The Mongo ``context`` collection; defaults to
                the configured context DAO's collection.

        Returns:
            Number of vectors written.

        Raises:
            IndexDimensionMismatchError: The rows hold embeddings of more than
                one dimension (nothing is written).
        """
        if context_collection is None:
            # Imported here: the DAO module imports this one.
            from src.rag_service.dao.factory import get_context_dao

            context_collection = get_context_dao().collection

        rows = context_collection.find(
            {"agent_id": agent_id, "embedding": {"$exists": True}}
        ).sort([("document_id", 1), ("chunk_index", 1)])

        store = FAISSVectorStore(agent_id, load_existing=False)
        for row in rows:
            embedding = row.get("embedding")
            if not embedding:
                continue
            document_id = row.get("document_id")
            # Same fallback as MongoDBContextDAO.insert_context for old rows.
            chunk_id = row.get("chunk_id") or f"{document_id}_{row.get('chunk_index')}"
            if chunk_id in store.chunk_metadata:
                # Metadata is keyed by chunk id; a second vector would shift
                # every later result. Rows from before unique ids may collide.
                logger.warning(
                    f"Skipping duplicate chunk id {chunk_id} for agent {agent_id}"
                )
                continue
            store.add_vectors(
                chunk_id,
                embedding,
                document_id,
                row.get("document_name"),
                row.get("text"),
                row.get("chunk_index"),
                row.get("total_chunks"),
                persist=False,
            )

        count = store.size()
        if count == 0:
            logger.warning(
                f"No context rows with embeddings for agent {agent_id}; "
                f"FAISS files left unchanged"
            )
            return 0

        with _agent_lock(agent_id):
            store._save_index()
            self._stores[agent_id] = store
        logger.info(
            f"Rebuilt FAISS index for agent {agent_id}: {count} vectors of "
            f"dimension {store.embedding_dim}"
        )
        return count

    def get_metadata(self, agent_id: str, chunk_id: str) -> dict | None:
        """Get metadata for a chunk.

        Args:
            agent_id: The agent ID
            chunk_id: The chunk ID

        Returns:
            Metadata dictionary or None
        """
        store = self._stores.get(agent_id)
        if store is not None:
            return store.get_metadata(chunk_id)
        return None
