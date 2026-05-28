"""Local FAISS-based vector store for self-contained vector search.

This module provides vector storage and retrieval using FAISS (Facebook AI Similarity Search)
with cosine similarity. It replaces MongoDB Atlas Vector Search for self-contained deployments.

Key Features:
- Local vector indexing with FAISS (no external dependencies)
- Cosine similarity search
- Per-agent vector indices (isolated namespaces)
- Persistence to disk
- Memory-efficient with dimension-aware index selection
"""

import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


logger = logging.getLogger(__name__)

# Directory for storing FAISS indices
FAISS_INDEX_DIR = Path("/data/faiss_indices")


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


class FAISSVectorStore:
    """Local vector store using FAISS for cosine similarity search.
    
    This class manages vector indices for individual agents, supporting
    efficient similarity search without external dependencies.
    """
    
    def __init__(self, agent_id: str, embedding_dim: int = 384):
        """Initialize FAISS vector store for an agent.
        
        Args:
            agent_id: The agent ID
            embedding_dim: Dimension of embeddings (default 384 for text-embedding-3-small)
        """
        self.agent_id = agent_id
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.chunk_metadata: dict = {}  # chunk_id -> {document_id, document_name, text, ...}
        self._load_or_create_index()
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create a new one.
        
        If an existing index has mismatched dimensions, it will be recreated.
        """
        index_path = get_index_path(self.agent_id)
        metadata_path = get_metadata_path(self.agent_id)
        
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                
                # Check if dimensions match
                if self.index.d != self.embedding_dim:
                    logger.warning(
                        f"FAISS index dimension mismatch for agent {self.agent_id}: "
                        f"index has {self.index.d}D but expected {self.embedding_dim}D. "
                        f"Recreating index with correct dimensions."
                    )
                    # Remove old index and metadata
                    index_path.unlink(missing_ok=True)
                    metadata_path.unlink(missing_ok=True)
                    self._create_new_index()
                else:
                    logger.info(f"Loaded FAISS index for agent {self.agent_id} with {self.index.ntotal} vectors")
                    
                    # Load metadata
                    if metadata_path.exists():
                        metadata = np.load(str(metadata_path), allow_pickle=True).item()
                        self.chunk_metadata = metadata
                        logger.info(f"Loaded metadata for {len(self.chunk_metadata)} chunks")
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index with cosine similarity."""
        # Use IndexFlatL2 with cosine distance
        # For cosine similarity, we normalize vectors and use L2 distance
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        logger.info(f"Created new FAISS index for agent {self.agent_id}")
    
    def add_vectors(
        self,
        chunk_id: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
        text: str,
        chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
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
        """
        if not self.index:
            self._create_new_index()
        
        # Validate embedding dimensions
        if len(embedding) != self.embedding_dim:
            logger.error(
                f"Embedding dimension mismatch for chunk {chunk_id}: "
                f"expected {self.embedding_dim}, got {len(embedding)}"
            )
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(embedding)}"
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
        """
        if not self.index or self.index.ntotal == 0:
            logger.warning(f"No vectors in index for agent {self.agent_id}")
            return {}
        
        try:
            # Log query details
            logger.debug(f"FAISS search for agent {self.agent_id}: query_embedding type={type(query_embedding)}, available_documents={available_documents}")
            
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
            
            # Validate query embedding dimensions
            if query_vector.shape[1] != self.embedding_dim:
                logger.error(
                    f"Query embedding dimension mismatch for agent {self.agent_id}: "
                    f"expected {self.embedding_dim}, got {query_vector.shape[1]}"
                )
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
            available_docs_set = set(available_documents) if available_documents else set()
            
            # Search - get more candidates to account for filtering
            search_k = min(top_k * 5, self.index.ntotal)
            distances, indices = self.index.search(query_vector, search_k)
            
            logger.debug(f"FAISS raw search for agent {self.agent_id}: index_size={self.index.ntotal}, search_k={search_k}, raw_results={len(indices[0])}")
            
            # Create chunk_id list in index order (must match insertion order)
            chunk_ids = list(self.chunk_metadata.keys())
            
            logger.debug(f"FAISS metadata for agent {self.agent_id}: chunk_ids_count={len(chunk_ids)}, available_docs={available_docs_set}")
            
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
                cosine_similarity = max(0.0, 1.0 - (dist_float ** 2 / 2.0))
                
                if not available_docs_set:
                    logger.debug(f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f} (no filter)")
                elif doc_id not in available_docs_set:
                    logger.debug(f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f} FILTERED OUT")
                    continue
                else:
                    logger.debug(f"  Result {i}: chunk={chunk_id}, doc={doc_id}, sim={cosine_similarity:.4f}")
                
                # Convert L2 distance to cosine similarity (0-1 range)
                # For L2 normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)
                
                # Apply threshold
                if cosine_similarity >= similarity_threshold:
                    results[chunk_id] = cosine_similarity
                else:
                    logger.debug(f"    Below threshold ({similarity_threshold})")
            
            # Sort by similarity and return top_k
            sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]
            logger.info(f"FAISS search for agent {self.agent_id}: found {len(sorted_results)} final results (after filtering)")
            return dict(sorted_results)
            
        except Exception as e:
            logger.error(f"Error in FAISS search for agent {self.agent_id}: {e}", exc_info=True)
            return {}
    
    def _save_index(self) -> None:
        """Persist the index to disk."""
        if not self.index:
            return
        
        index_path = get_index_path(self.agent_id)
        metadata_path = get_metadata_path(self.agent_id)
        
        try:
            faiss.write_index(self.index, str(index_path))
            np.save(str(metadata_path), self.chunk_metadata)
            logger.debug(f"Saved FAISS index for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
    
    def get_metadata(self, chunk_id: str) -> Optional[dict]:
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
    
    _instance: Optional['FAISSVectorStoreManager'] = None
    _stores: dict[str, FAISSVectorStore] = {}
    
    def __new__(cls) -> 'FAISSVectorStoreManager':
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_store(self, agent_id: str, embedding_dim: int = 384) -> FAISSVectorStore:
        """Get or create a vector store for an agent.
        
        Args:
            agent_id: The agent ID
            embedding_dim: Dimension of embeddings
            
        Returns:
            FAISSVectorStore instance for the agent
        """
        if agent_id not in self._stores:
            self._stores[agent_id] = FAISSVectorStore(agent_id, embedding_dim)
        return self._stores[agent_id]
    
    def add_vector(
        self,
        agent_id: str,
        chunk_id: str,
        embedding: list[float],
        document_id: str,
        document_name: str,
        text: str,
        chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        embedding_dim: int = 384,
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
            embedding_dim: Dimension of embeddings (default 768)
        """
        logger.info(f"FAISSVectorStoreManager.add_vector: agent={agent_id}, chunk={chunk_id}, embedding_dim={embedding_dim} (actual={len(embedding)}), doc={document_name}")
        
        store = self.get_store(agent_id, embedding_dim)
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
        embedding_dim: int = 384,
    ) -> dict[str, float]:
        """Search in an agent's vector store.
        
        Args:
            agent_id: The agent ID
            query_embedding: Query embedding
            available_documents: Documents to filter by
            top_k: Number of results
            similarity_threshold: Minimum similarity score
            embedding_dim: Dimension of embeddings
            
        Returns:
            Dictionary mapping chunk_id -> similarity_score
        """
        store = self.get_store(agent_id, embedding_dim)
        return store.search(
            query_embedding,
            available_documents,
            top_k,
            similarity_threshold,
        )
    
    def get_metadata(self, agent_id: str, chunk_id: str) -> Optional[dict]:
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
