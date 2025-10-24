import logging

from src.rag_service.context import Context
from src.rag_service.embeddings import similarity_search

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

    all_doc_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    hybrid_scores = {}
    for doc_id in all_doc_ids:
        vector_score = vector_scores.get(doc_id, 0)
        keyword_score = keyword_scores.get(doc_id, 0)
        hybrid_scores[doc_id] = alpha * vector_score + (1 - alpha) * keyword_score

    top_indices = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)[:top_k]
    
    top_indices = [
        idx for idx in top_indices if hybrid_scores.get(idx, 0) >= similarity_threshold
    ]
    
    results = []
    for doc_id in top_indices:
        document = document_map.get(doc_id)
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

def keyword_search(agent_id, keyword_query_text, available_documents, context_collection, top_k, document_map):
    keyword_scores = {}
    keyword_pipeline = [
            {
                "$search": {
                    "index": "keyword_search",
                    "compound": {
                        "must": [
                            {
                                "text": {
                                    "query": keyword_query_text,
                                    "path": "text"
                                }
                            }
                        ],
                        "filter": [
                            {
                                "equals": {
                                    "path": "agent_id",
                                    "value": agent_id
                                }
                            },
                            {
                                "in": {
                                    "path": "document_id",
                                    "value": available_documents
                                }
                            }
                        ]
                    }
                }
            },
            {
                "$limit": top_k * 3
            },
            {
                "$addFields": {
                    "keyword_score": {"$meta": "searchScore"}
                }
            }
        ]

    keyword_results = list(context_collection.aggregate(keyword_pipeline))

    for doc in keyword_results:
        doc_id = doc["document_id"]
        keyword_scores[doc_id] = doc.get("keyword_score", 0)
        document_map[doc_id] = doc

    return keyword_scores

def vector_search(agent_id, embedded_query, available_documents, context_collection, num_candidates, top_k, document_map):
    vector_scores = {}
    search_filter = {"agent_id": {"$eq": agent_id}}
    search_filter["document_id"] = {"$in": available_documents}

    # Ensure limit does not exceed numCandidates (MongoDB Atlas requirement)
    limit = min(top_k * 3, num_candidates)

    vector_pipeline = [
            {
                "$vectorSearch": {
                    "index": "embeddings",
                    "path": "embedding",
                    "queryVector": embedded_query,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "filter": search_filter,
                }
            },
            {
                "$addFields": {
                    "vector_score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

    vector_results = list(context_collection.aggregate(vector_pipeline))

    for doc in vector_results:
        doc_id = doc["document_id"]
        vector_scores[doc_id] = doc.get("vector_score", 0)
        document_map[doc_id] = doc

    return vector_scores
