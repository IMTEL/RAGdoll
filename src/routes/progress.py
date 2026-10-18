import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import ValidationError

from src.config import Config
from src.globals import access_service, agent_dao, progress_dao
from src.models.training import ListProgressData, ProgressData


# Define a router for progress-related endpoints
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/progress", tags=["Progress"])
DEFAULT_SESSION_ID = "default"


def _normalize_session_id(session_id: str | None) -> str:
    normalized = session_id.strip() if session_id else ""
    return normalized or DEFAULT_SESSION_ID


def _authorize_agent_access(agent_id: str | None, access_key: str | None) -> str:
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")
    if not access_key:
        raise HTTPException(status_code=401, detail="access_key is required")

    agent = agent_dao.get_agent_by_id(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404, detail=f"Agent with id '{agent_id}' not found"
        )
    if not access_service.authenticate(agent_id, access_key):
        raise HTTPException(status_code=401, detail="Unauthorized access to agent")
    return agent_id


def _progress_dump(progress: ProgressData, agent_id: str, session_id: str) -> dict:
    entry = progress.model_dump(exclude={"access_key", "completet_at"})
    entry["agent_id"] = agent_id
    entry["session_id"] = session_id
    entry["updated_at"] = datetime.now(UTC)
    return entry


def get_recent_progress_for_session(
    agent_id: str, session_id: str | None, limit: int = 5
) -> list[ProgressData]:
    if limit <= 0:
        return []

    entries = progress_dao.list_session_progress(
        agent_id, _normalize_session_id(session_id), limit
    )
    return [ProgressData.model_validate(entry) for entry in entries]


@router.post("/initializeTasks")
def receive_hierarchy(task_hierarchy: ListProgressData):
    """Initializes a list of tasks with their subtasks and steps."""
    try:
        agent_id = _authorize_agent_access(
            task_hierarchy.agent_id, task_hierarchy.access_key
        )
        session_id = _normalize_session_id(task_hierarchy.session_id)
        for task in task_hierarchy.items:
            progress_dao.add_progress(
                _progress_dump(
                    task,
                    agent_id,
                    _normalize_session_id(task.session_id or session_id),
                )
            )
        return {
            "message": "Tasks initialized",
            "data": task_hierarchy.model_dump(
                exclude={"access_key": True, "items": {"__all__": {"access_key"}}}
            ),
        }
    except Exception as e:
        if isinstance(e, ValidationError):
            logger.warning(
                "Error processing task hierarchy: ValidationError (%d errors)",
                e.error_count(),
            )
        else:
            logger.warning("Error processing task hierarchy: %s", type(e).__name__)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/session")
def create_progress_session(
    agent_id: str,
    access_key: Annotated[str | None, Header()] = None,
):
    """Create a backend-generated session id for external progress tracking."""
    _authorize_agent_access(agent_id, access_key)
    return {
        "agent_id": agent_id,
        "session_id": f"ragdoll-session-{uuid.uuid4()}",
        "expires_after_hours": int(Config().PROGRESS_TTL_HOURS),
    }


@router.post("/updateTask")
def receive_progress(progress: ProgressData):
    """Handles task progress updates (started/complete) and stores them."""
    agent_id = _authorize_agent_access(progress.agent_id, progress.access_key)
    session_id = _normalize_session_id(progress.session_id)

    if progress.status == "started" or progress.status == "pending":
        now = datetime.now(UTC)
        entry = progress_dao.update_active_progress(
            agent_id,
            session_id,
            progress.task_name,
            {
                "subtask_progress": [
                    subtask.model_dump() for subtask in progress.subtask_progress
                ],
                "updated_at": now,
                "status": progress.status,
                "description": progress.description,
            },
        )
        if entry is not None:
            if not entry.get("started_at"):
                entry = progress_dao.update_active_progress(
                    agent_id,
                    session_id,
                    progress.task_name,
                    {"started_at": now},
                )
            return {"message": "Progress updated", "data": entry}

        # New task entry
        new_entry = _progress_dump(progress, agent_id, session_id)
        new_entry["started_at"] = datetime.now(UTC)
        return {
            "message": "Progress received",
            "data": progress_dao.add_progress(new_entry),
        }

    elif progress.status == "complete":
        now = datetime.now(UTC)
        entry = progress_dao.update_active_progress(
            agent_id,
            session_id,
            progress.task_name,
            {
                "subtask_progress": [
                    subtask.model_dump() for subtask in progress.subtask_progress
                ],
                "completed_at": now,
                "updated_at": now,
                "status": "complete",
                "description": progress.description,
            },
        )
        if entry is not None:
            return {"message": "Task completed", "data": entry}
        return {"message": f"No active task {progress.task_name} found."}

    else:
        raise HTTPException(400, "Status must be 'started', 'pending', or 'complete'.")


@router.get("")
def get_progress_log(
    agent_id: str,
    session_id: str | None = None,
    limit: int | None = None,
    access_key: Annotated[str | None, Header()] = None,
):
    authorized_agent_id = _authorize_agent_access(agent_id, access_key)
    return progress_dao.list_session_progress(
        authorized_agent_id, _normalize_session_id(session_id), limit
    )
