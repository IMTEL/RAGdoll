from fastapi import APIRouter, HTTPException
from src.models.rag import RAGPostModel
from src.rag_service.dao import get_database

router = APIRouter()

@router.post("/rag/post")
def post_rag_item(payload: RAGPostModel):
    """
    Store a new RAG item in the mock (or real) DB.
    """
    db = get_database()  # returns MockDatabase if config says "mock"
    try:
        success = db.post_context(
            text=payload.text,
            NPC=payload.NPC,
            embedding=payload.embedding,
            document_id=payload.document_id,
            document_name=payload.document_name,
        )
        if success:
            return {"message": "RAG context posted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store RAG context")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rag/data")
def get_rag_data():
    db = get_database()
    if hasattr(db, "data"):
        return db.data
    return []