from src.rag_service.dao.context import hybrid_search as hybrid_search_module


def test_hybrid_search_keeps_strong_keyword_match_when_vector_score_is_weak(
    monkeypatch,
):
    document = {
        "text": "Dinosaurs first appeared during the Triassic Period.",
        "document_name": "one_page_about_dinosaurs.pdf",
        "document_id": "doc-dino",
        "chunk_id": "chunk-dino",
        "chunk_index": 0,
        "total_chunks": 1,
    }

    def fake_vector_search(*args, **kwargs):
        return {"chunk-dino": 0.0}

    def fake_keyword_search(
        agent_id,
        keyword_query_text,
        available_documents,
        context_collection,
        top_k,
        document_map,
    ):
        document_map["chunk-dino"] = document
        return {"chunk-dino": 1.0}

    monkeypatch.setattr(hybrid_search_module, "vector_search", fake_vector_search)
    monkeypatch.setattr(hybrid_search_module, "keyword_search", fake_keyword_search)

    results = hybrid_search_module.hybrid_search(
        alpha=0.75,
        agent_id="agent-1",
        embedded_query=[0.1, 0.2],
        query_text="tell me about dinosaurs",
        keyword_query_text="tell me about dinosaurs",
        available_documents=["doc-dino"],
        context_collection=None,
        similarity_threshold=0.5,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].document_id == "doc-dino"
