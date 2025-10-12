from fastapi import APIRouter

from src.utils.global_logs import failure_log, progress_log


router = APIRouter()


@router.get("/api/debug/logs")
def get_logs():
    """Returns the in-memory logs for progress and failure."""
    return {"progressLog": progress_log, "failureLog": failure_log}
