"""Local FAISS-based vector store for self-contained vector search.

This module provides vector storage and retrieval using FAISS (Facebook AI Similarity Search)
with cosine similarity. It replaces MongoDB Atlas Vector Search for self-contained deployments.

Key Features:
- Local vector indexing with FAISS (no external dependencies)
- Cosine similarity search
- Per-agent vector indices (isolated namespaces)
- Persistence to disk (atomic writes; files are never deleted on load errors)
- The index dimension comes from the file, or from the first vector added
- Rebuild from the MongoDB context rows (see scripts/rebuild_faiss_index.py)
"""

import logging
import os
import pickle
import tempfile
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
        self.index: faiss.Index | None = None
        self.chunk_metadata: dict = {}  # chunk_id -> {document_id, document_name, text, ...}
        # Identity of the files this copy matches (from the last load or save).
        # Another process (the rebuild CLI, another worker) may replace them;
        # refresh() then reloads instead of serving or overwriting stale data.
        # A store started with load_existing=False ignores the disk until it
        # saves for the first time.
        self._sync_with_disk = load_existing
        self._disk_signature: tuple | None = None
        if load_existing:
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
        to match the disk.
        """
        if not self._sync_with_disk:
            return
        if self._current_disk_signature() == self._disk_signature:
            return
        logger.info(
            f"FAISS index files for agent {self.agent_id} changed on disk; reloading"
        )
        self.index = None
        self.chunk_metadata = {}
        self._load_index(None)

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
                index, metadata = self._read_files(index_path, metadata_path)
            except _READ_ERRORS:
                # Another process may be between replacing the index and the
                # metadata file; give it a moment before calling it corrupt.
                time.sleep(0.2)
                signature = self._current_disk_signature()
                index, metadata = self._read_files(index_path, metadata_path)
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

        self.index = index
        self.chunk_metadata = metadata
        self._disk_signature = signature
        logger.info(
            f"Loaded FAISS index for agent {self.agent_id} with {index.ntotal} "
            f"vectors of dimension {index.d}"
        )
        self.check_dimension(requested_dim)

    @staticmethod
    def _read_files(index_path: Path, metadata_path: Path) -> tuple:
        index = faiss.read_index(str(index_path))
        metadata = (
            np.load(str(metadata_path), allow_pickle=True).item()
            if metadata_path.exists()
            else {}
        )
        if not isinstance(metadata, dict):
            raise TypeError("metadata file does not hold a dict")
        # Search maps result positions to metadata order, so they must agree.
        if len(metadata) != index.ntotal:
            raise ValueError(
                f"metadata has {len(metadata)} entries but the index has "
                f"{index.ntotal} vectors"
            )
        return index, metadata

    def _create_new_index(self, embedding_dim: int) -> None:
        """Create a new FAISS index with cosine similarity."""
        # Use IndexFlatL2 with cosine distance
        # For cosine similarity, we normalize vectors and use L2 distance
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(
            f"Created new {embedding_dim}-dim FAISS index for agent {self.agent_id}"
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
        self.refresh()
        if chunk_id in self.chunk_metadata:
            # Metadata is keyed by chunk id; a second vector under the same id
            # would desync index and metadata (and every later result).
            raise FAISSIndexError(
                f"Chunk {chunk_id} is already in the FAISS index for agent "
                f"{self.agent_id}"
            )
        if self.index is None:
            self._create_new_index(len(embedding))

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

        # Add to FAISS index
        self.index.add(vector)
        size_after = self.index.ntotal

        # Store metadata
        self.chunk_metadata[chunk_id] = {
            "document_id": document_id,
            "document_name": document_name,
            "text": text,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }

        # Save index and metadata
        if persist:
            self._save_index()

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
            similarity_threshold: Minimum similarity score (0-1 for cosine)

        Returns:
            Dictionary mapping chunk_id -> similarity_score (0-1 range)

        Raises:
            IndexDimensionMismatchError: The query has another dimension.
        """
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

            # Create chunk_id list in index order (must match insertion order)
            chunk_ids = list(self.chunk_metadata.keys())

            logger.debug(
                f"FAISS metadata for agent {self.agent_id}: chunk_ids_count={len(chunk_ids)}, available_docs={available_docs_set}"
            )

            results = {}
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                idx = int(idx)
                if idx < 0 or idx >= len(chunk_ids):
                    logger.debug(f"  Result {i}: invalid index {idx}")
                    continue

                chunk_id = chunk_ids[idx]
                metadata = self.chunk_metadata.get(chunk_id, {})

                # Filter by available documents
                doc_id = metadata.get("document_id")
                dist_float = float(dist)
                cosine_similarity = max(0.0, 1.0 - (dist_float**2 / 2.0))

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

                # Convert L2 distance to cosine similarity (0-1 range)
                # For L2 normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)

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
        """
        if not self.index:
            return

        index_path = get_index_path(self.agent_id)
        metadata_path = get_metadata_path(self.agent_id)

        try:
            _atomic_write(
                index_path, lambda tmp: faiss.write_index(self.index, str(tmp))
            )
            _atomic_write(
                metadata_path, lambda tmp: _save_metadata(tmp, self.chunk_metadata)
            )
            self._disk_signature = self._current_disk_signature()
            self._sync_with_disk = True
            logger.debug(f"Saved FAISS index for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index for agent {self.agent_id}: {e}")
            raise FAISSIndexError(
                f"Failed to save FAISS index for agent {self.agent_id}: {e}"
            ) from e

    def get_metadata(self, chunk_id: str) -> dict | None:
        """Get metadata for a chunk.

        Args:
            chunk_id: The chunk ID

        Returns:
            Metadata dictionary or None if not found
        """
        return self.chunk_metadata.get(chunk_id)

    def size(self) -> int:
        """Get the number of vectors in the index."""
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

    def rebuild_index(self, agent_id: str, context_collection=None) -> int:
        """Rebuild an agent's FAISS index from its MongoDB context rows.

        The rows store each chunk's embedding, so MongoDB can always recreate
        the index. The new index is built in memory through the normal add
        path and only replaces the files on disk once every row was added.
        With no rows, existing files are left alone.

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
        if agent_id in self._stores:
            return self._stores[agent_id].get_metadata(chunk_id)
        return None
