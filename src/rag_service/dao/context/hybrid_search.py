import re
from rank_bm25 import BM25Okapi

from src.rag_service.context import Context
from src.rag_service.embeddings import similarity_search
import logging

logger = logging.getLogger(__name__)

def hybrid_search(alpha: float, # 0 = 100% keyword, 1 = 100% vector, 0.5 = equal weight
                  agent_id: str,
                  embedded_query: list[float], 
                  query_text: str,
                  keyword_query_text: str,
                  available_documents: list[str],
                  context_collection,
                  similarity_threshold: float,
                  num_candidates: int = 50, 
                  top_k: int = 5) -> list[Context]:

    # do semantic search
    search_filter = {"agent_id": {"$eq": agent_id}}
    search_filter["document_id"] = {"$in": available_documents}
    pipeline = [
        {
            "$vectorSearch": {
                "index": "embeddings",
                "path": "embedding",
                "queryVector": embedded_query,
                "numCandidates": num_candidates,
                "limit": top_k * 10, # get more to filter later
                "filter": search_filter,
            }
        },
    ]

    # hybrid search inspired by https://www.machinelearningplus.com/gen-ai/hybrid-search-vector-keyword-techniques-for-better-rag/

    vector_results = []
    tokenized_corpus = []
    document_map = {}
    for doc in context_collection.aggregate(pipeline):
        similarity = similarity_search(embedded_query, doc["embedding"])
        vector_results.append((doc["document_id"], similarity))
        tokenized_corpus.append(preprocess_text_for_bm25(doc["text"]))
        document_map[doc["document_id"]] = doc

    keyword_searcher = BM25Okapi(tokenized_corpus) # store in database
    
    # Use separate query for keyword search (typically just the user's question)
    preprocessed_query = preprocess_text_for_bm25(keyword_query_text)
    bm25_scores_list = keyword_searcher.get_scores(preprocessed_query)

    vector_scores = {}
    for doc, score in vector_results:
        # similarity_search returns cosine similarity (range -1 to 1, where 1 is most similar)
        # Normalize to 0-1 range: (score + 1) / 2
        normalized_score = (score + 1) / 2

        vector_scores[doc] = normalized_score

    bm25_scores = {}
    # Use min-max normalization to handle negative BM25 scores
    min_score = float(min(bm25_scores_list)) if len(bm25_scores_list) > 0 else 0.0
    max_score = float(max(bm25_scores_list)) if len(bm25_scores_list) > 0 else 1.0
    score_range = max_score - min_score
    
    for idx, score in enumerate(bm25_scores_list):
        # Min-max normalization: (score - min) / (max - min)
        normalized_score = float((score - min_score) / score_range) if score_range > 0 else 0.0
        bm25_scores[vector_results[idx][0]] = normalized_score

    hybrid_scores = {}
    all_indices = set(vector_scores.keys())

    for idx in all_indices:
        vector_score = vector_scores.get(idx, 0)
        bm25_score = bm25_scores.get(idx, 0)
        hybrid_scores[idx] = alpha * vector_score + (1 - alpha) * bm25_score

    print(f"Hybrid scores: {hybrid_scores}, Vector scores: {vector_scores}, BM25 scores: {bm25_scores}")

    top_indices = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:top_k]
    top_indices = [
        idx for idx in top_indices if hybrid_scores.get(idx, 0) >= similarity_threshold
    ]
    
    results = []
    for id in top_indices:
        document = document_map.get(id)
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


def preprocess_text_for_bm25(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = [token for token in text.split() if token.strip()]
    return tokens
