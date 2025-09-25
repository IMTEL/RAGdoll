from fastapi import APIRouter
from src.utils.global_logs import progressLog, failureLog

router = APIRouter()


@router.get("/api/debug/logs")
def get_logs():
    """
    Returns the in-memory logs for progress and failure.
    """
    return {"progressLog": progressLog, "failureLog": failureLog}
