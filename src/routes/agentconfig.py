from src.LLM import get_models
from fastapi import APIRouter
from src.utils.global_logs import progressLog, failureLog

router = APIRouter()

@router.get("/api/agentconfig/models")
def fetch_models():
    """
    Returns all usable models.
    """
    return get_models()
    
