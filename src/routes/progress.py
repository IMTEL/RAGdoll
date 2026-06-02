from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from src.globals import access_service, agent_dao
from src.models.training import ListProgressData, ProgressData

# In-memory log to store progress data
from src.utils.global_logs import progress_log


# Define a router for progress-related endpoints
router = APIRouter()
DEFAULT_SESSION_ID = "default"
PROGRESS_TTL = timedelta(hours=24)


def _normalize_session_id(session_id: str | None) -> str:
    normalized = session_id.strip() if session_id else ""
    return normalized or DEFAULT_SESSION_ID


def _entry_timestamp(entry: dict) -> datetime:
    timestamp = entry.get("updated_at") or entry.get("started_at") or datetime.min
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp


def _prune_expired_progress() -> None:
    cutoff = datetime.now(UTC) - PROGRESS_TTL
    progress_log[:] = [
        entry for entry in progress_log if _entry_timestamp(entry) >= cutoff
    ]


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
    entry = progress.model_dump(exclude={"access_key"})
    entry["agent_id"] = agent_id
    entry["session_id"] = session_id
    entry["updated_at"] = datetime.now(UTC)
    return entry


def get_recent_progress_for_session(
    agent_id: str, session_id: str | None, limit: int = 5
) -> list[ProgressData]:
    if limit <= 0:
        return []

    _prune_expired_progress()
    normalized_session_id = _normalize_session_id(session_id)
    entries = [
        entry
        for entry in progress_log
        if entry.get("agent_id") == agent_id
        and entry.get("session_id", DEFAULT_SESSION_ID) == normalized_session_id
    ]
    entries.sort(key=_entry_timestamp, reverse=True)
    return [
        ProgressData.model_validate(entry)
        for entry in entries[:limit]
    ]


@router.post("/api/progress/initializeTasks")
def receive_hierarchy(task_hierarchy: ListProgressData):
    """Initializes a list of tasks with their subtasks and steps."""
    try:
        _prune_expired_progress()
        agent_id = _authorize_agent_access(
            task_hierarchy.agent_id, task_hierarchy.access_key
        )
        session_id = _normalize_session_id(task_hierarchy.session_id)
        for task in task_hierarchy.items:
            progress_log.append(
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
        print(f"Error processing task hierarchy: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/progress/updateTask")
def receive_progress(progress: ProgressData):
    """Handles task progress updates (started/complete) and stores them."""
    _prune_expired_progress()
    agent_id = _authorize_agent_access(progress.agent_id, progress.access_key)
    session_id = _normalize_session_id(progress.session_id)

    if progress.status == "started" or progress.status == "pending":
        # Check for existing incomplete task to update
        for entry in progress_log:
            if (
                entry.get("agent_id") == agent_id
                and entry.get("session_id", DEFAULT_SESSION_ID) == session_id
                and entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                now = datetime.now(UTC)
                entry.update(
                    {
                        "subtask_progress": [
                            subtask.model_dump()
                            for subtask in progress.subtask_progress
                        ],
                        "started_at": entry.get("started_at") or now,
                        "updated_at": now,
                        "status": progress.status,
                        "description": progress.description,
                    }
                )
                return {"message": "Progress updated", "data": entry}

        # New task entry
        new_entry = _progress_dump(progress, agent_id, session_id)
        new_entry["started_at"] = datetime.now(UTC)
        progress_log.append(new_entry)
        return {"message": "Progress received", "data": new_entry}

    elif progress.status == "complete":
        # Complete existing task
        for entry in progress_log:
            if (
                entry.get("agent_id") == agent_id
                and entry.get("session_id", DEFAULT_SESSION_ID) == session_id
                and entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                now = datetime.now(UTC)
                entry.update(
                    {
                        "subtask_progress": [
                            subtask.model_dump()
                            for subtask in progress.subtask_progress
                        ],
                        "completed_at": now,
                        "completet_at": now,
                        "updated_at": now,
                        "status": "complete",
                        "description": progress.description,
                    }
                )
                return {"message": "Task completed", "data": entry}
        return {"message": f"No active task {progress.task_name} found."}

    else:
        raise HTTPException(400, "Status must be 'started', 'pending', or 'complete'.")


@router.get("/api/progress")
def get_progress_log(
    agent_id: str,
    session_id: str | None = None,
    limit: int | None = None,
    access_key: Annotated[str | None, Header()] = None,
):
    _prune_expired_progress()
    authorized_agent_id = _authorize_agent_access(agent_id, access_key)
    normalized_session_id = _normalize_session_id(session_id)
    entries = [
        entry
        for entry in progress_log
        if entry.get("agent_id") == authorized_agent_id
        and entry.get("session_id", DEFAULT_SESSION_ID) == normalized_session_id
    ]
    entries.sort(key=_entry_timestamp, reverse=True)
    return entries[:limit] if limit is not None and limit >= 0 else entries
