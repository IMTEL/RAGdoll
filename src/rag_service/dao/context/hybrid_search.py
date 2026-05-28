import logging
import re
from collections import defaultdict

from pymongo.errors import PyMongoError

from src.rag_service.context import Context
from src.rag_service.dao.context.faiss_vector_store import FAISSVectorStoreManager


logger = logging.getLogger(__name__)

_INDEX_CHECK_CACHE: set[str] = set()


def _debug_log_search_index(collection, index_name: str) -> None:
    """Log whether the expected search index exists for debugging purposes."""
    cache_key = f"{id(collection)}::{index_name}"
    if cache_key in _INDEX_CHECK_CACHE:
        return

    if not hasattr(collection, "list_search_indexes"):
        logger.debug(
            "Collection does not support list_search_indexes; cannot verify '%s' index.",
            index_name,
        )
        _INDEX_CHECK_CACHE.add(cache_key)
        return

    try:
        try:
            matching_indexes = list(collection.list_search_indexes(name=index_name))
        except TypeError:
            matching_indexes = [
                idx
                for idx in collection.list_search_indexes()
                if idx.get("name") == index_name
            ]

        if matching_indexes:
            logger.debug("Confirmed search index '%s' exists.", index_name)
        else:
            logger.warning(
                "Search index '%s' not found; keyword search may not work optimally.",
                index_name,
            )
    except PyMongoError as exc:
        logger.debug("Could not verify search index '%s': %s (This is normal for Community MongoDB)", index_name, exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(
            "Unexpected error while verifying search index '%s': %s", index_name, exc
        )
    finally:
        _INDEX_CHECK_CACHE.add(cache_key)


def hybrid_search(
    alpha: float,  # 0 = 100% keyword, 1 = 100% vector, 0.5 = equal weight
    agent_id: str,
    embedded_query: list[float],
    query_text: str,
    keyword_query_text: str,
    available_documents: list[str],
    context_collection,
    similarity_threshold: float,
    num_candidates: int = 50,
    top_k: int = 5,
) -> list[Context]:
    document_map = {}
    vector_scores = {}

    if alpha > 0:
        vector_scores = vector_search(
            agent_id,
            embedded_query,
            available_documents,
            context_collection,
            num_candidates,
            top_k,
            document_map,
        )
    else:
        vector_scores = {}

    if alpha < 1:
        keyword_scores = keyword_search(
            agent_id,
            keyword_query_text,
            available_documents,
            context_collection,
            top_k,
            document_map,
        )
    else:
        keyword_scores = {}

    print(vector_scores, keyword_scores)

    all_chunk_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    hybrid_scores = {}
    for chunk_id in all_chunk_ids:
        vector_score = vector_scores.get(chunk_id, 0)
        keyword_score = keyword_scores.get(chunk_id, 0)
        hybrid_scores[chunk_id] = alpha * vector_score + (1 - alpha) * keyword_score

    top_indices = sorted(
        hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True
    )[:top_k]

    top_indices = [
        idx for idx in top_indices if hybrid_scores.get(idx, 0) >= similarity_threshold
    ]

    results = []
    for chunk_id in top_indices:
        document = document_map.get(chunk_id)
        if document:
            results.append(
                Context(
                    text=document["text"],
                    document_name=document["document_name"],
                    document_id=document.get("document_id"),
                    chunk_id=document.get("chunk_id"),
                    chunk_index=document.get("chunk_index"),
                    total_chunks=document.get("total_chunks", 1),
                )
            )

    return results


def keyword_search(
    agent_id,
    keyword_query_text,
    available_documents,
    context_collection,
    top_k,
    document_map,
):
    """Local BM25-based keyword search for MongoDB Community Edition.
    
    Uses simple tokenization and BM25 scoring without requiring MongoDB Atlas.
    Results are cached per query to improve performance.
    
    Args:
        agent_id: Agent ID to filter by
        keyword_query_text: Query text to search for
        available_documents: List of document IDs to filter by
        context_collection: MongoDB collection
        top_k: Number of results to return
        document_map: Output dict to populate with documents
        
    Returns:
        Dictionary mapping chunk_id -> relevance score (0-1 range)
    """
    keyword_scores = {}
    
    if not keyword_query_text or not keyword_query_text.strip():
        return keyword_scores
    
    try:
        # Fetch all chunks for the available documents from MongoDB
        query = {
            "agent_id": agent_id,
            "document_id": {"$in": available_documents}
        }
        
        documents = list(context_collection.find(query))
        if not documents:
            logger.debug(f"No documents found for keyword search (agent={agent_id})")
            return keyword_scores
        
        # Tokenize query (simple lowercase split on whitespace and punctuation)
        query_tokens = _tokenize(keyword_query_text)
        if not query_tokens:
            return keyword_scores
        
        logger.debug(f"Keyword search for agent {agent_id}: query_tokens={query_tokens}, num_docs={len(documents)}")
        
        # Calculate BM25 scores for each document
        bm25_scores = _calculate_bm25_scores(
            query_tokens,
            documents,
            k1=1.5,  # BM25 parameter (term frequency saturation)
            b=0.75   # BM25 parameter (document length normalization)
        )
        
        # Convert to chunk_id mapping and filter by top_k
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 3]
        
        for chunk_id, score in sorted_scores:
            # Find the corresponding document to populate document_map
            doc = next((d for d in documents if d.get("chunk_id", str(d.get("_id"))) == chunk_id), None)
            if doc:
                keyword_scores[chunk_id] = score
                document_map[chunk_id] = doc
        
        logger.debug(f"Keyword search found {len(keyword_scores)} results")
        
    except Exception as e:
        logger.warning(f"Error in local keyword search: {e}")
        # Return empty results on error - hybrid search will use vector results only
        keyword_scores = {}
    
    return keyword_scores


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, remove punctuation, split on whitespace.
    
    Args:
        text: Text to tokenize
        
    Returns:
        List of tokens (words)
    """
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split on whitespace and filter empty tokens
    tokens = [t for t in text.split() if t]
    return tokens


def _calculate_bm25_scores(
    query_tokens: list[str],
    documents: list[dict],
    k1: float = 1.5,
    b: float = 0.75
) -> dict[str, float]:
    """Calculate BM25 scores for each document.
    
    BM25 is a ranking function used in information retrieval. It considers:
    - Term frequency in document
    - Inverse document frequency (rarity across corpus)
    - Document length normalization
    
    Args:
        query_tokens: List of query tokens
        documents: List of document dicts with 'text' and 'chunk_id' fields
        k1: Term frequency saturation parameter (typically 1.5)
        b: Document length normalization parameter (typically 0.75)
        
    Returns:
        Dictionary mapping chunk_id -> BM25 score (0-1 range)
    """
    if not documents or not query_tokens:
        return {}
    
    # Build corpus statistics
    corpus_size = len(documents)
    avg_doc_length = sum(len(_tokenize(doc.get("text", ""))) for doc in documents) / corpus_size
    
    # Calculate document frequencies for each query term
    doc_freq = defaultdict(int)
    doc_tokens_cache = {}  # Cache tokenized docs for reuse
    
    for doc in documents:
        tokens = _tokenize(doc.get("text", ""))
        chunk_id = doc.get("chunk_id", str(doc.get("_id")))
        doc_tokens_cache[chunk_id] = tokens
        
        # Count which query terms appear in this doc
        unique_query_terms = set(query_tokens)
        for term in unique_query_terms:
            if term in tokens:
                doc_freq[term] += 1
    
    # Calculate BM25 score for each document
    scores = {}
    
    for doc in documents:
        chunk_id = doc.get("chunk_id", str(doc.get("_id")))
        tokens = doc_tokens_cache[chunk_id]
        doc_length = len(tokens)
        
        # Calculate term frequencies in this document
        term_freqs = defaultdict(int)
        for token in tokens:
            term_freqs[token] += 1
        
        # Sum BM25 component for each query term
        score = 0.0
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            tf = term_freqs[term]
            df = doc_freq[term]
            
            # Avoid log(0) - IDF is at least log(1)
            idf = max(0, (corpus_size - df + 0.5) / (df + 0.5))
            
            # BM25 formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / (avg_doc_length + 1e-6)))
            score += idf * (numerator / (denominator + 1e-6))
        
        scores[chunk_id] = score
    
    # Normalize to 0-1 range
    max_score = max(scores.values()) if scores.values() else 1.0
    if max_score > 0:
        scores = {chunk_id: score / max_score for chunk_id, score in scores.items()}
    
    return scores


def vector_search(
    agent_id,
    embedded_query,
    available_documents,
    context_collection,
    num_candidates,
    top_k,
    document_map,
):
    """Search using FAISS local vector store with cosine similarity.
    
    Args:
        agent_id: Agent ID for vector store lookup
        embedded_query: Query embedding vector
        available_documents: List of document IDs to filter by
        context_collection: MongoDB collection (used for metadata lookups)
        num_candidates: Number of candidates to consider
        top_k: Number of results to return
        document_map: Output dictionary to populate with document metadata
        
    Returns:
        Dictionary mapping chunk_id -> cosine similarity score (0-1)
    """
    vector_scores = {}
    
    try:
        # Use FAISS vector store for similarity search
        manager = FAISSVectorStoreManager()
        vector_scores = manager.search(
            agent_id,
            embedded_query,
            available_documents,
            top_k=top_k,
            similarity_threshold=0.0,  # Will be filtered later
        )
        
        # Populate document_map with metadata from MongoDB
        for chunk_id in vector_scores.keys():
            try:
                doc = context_collection.find_one({"chunk_id": chunk_id})
                if doc:
                    document_map[chunk_id] = doc
                else:
                    # Fall back to FAISS metadata if not in MongoDB
                    metadata = manager.get_metadata(agent_id, chunk_id)
                    if metadata:
                        document_map[chunk_id] = {
                            **metadata,
                            "chunk_id": chunk_id,
                        }
            except Exception as e:
                logger.debug(f"Could not find document for chunk {chunk_id}: {e}")
        
        logger.debug(f"Vector search found {len(vector_scores)} results for agent {agent_id}")
        
    except Exception as e:
        logger.error(f"Error retrieving vector search results: {e}")
        # Return empty results on error
        vector_scores = {}
    
    return vector_scores
