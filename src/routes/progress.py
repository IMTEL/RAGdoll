from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException

from src.globals import access_service, agent_dao
from src.models.training import ListProgressData, ProgressData

# In-memory log to store progress data
from src.utils.global_logs import progress_log


# Define a router for progress-related endpoints
router = APIRouter()


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


def _progress_dump(progress: ProgressData, agent_id: str) -> dict:
    entry = progress.model_dump(exclude={"access_key"})
    entry["agent_id"] = agent_id
    return entry


@router.post("/api/progress/initializeTasks")
def receive_hierarchy(task_hierarchy: ListProgressData):
    """Initializes a list of tasks with their subtasks and steps."""
    try:
        agent_id = _authorize_agent_access(
            task_hierarchy.agent_id, task_hierarchy.access_key
        )
        for task in task_hierarchy.items:
            progress_log.append(_progress_dump(task, agent_id))
        return {
            "message": "Tasks initialized",
            "data": task_hierarchy.model_dump(exclude={"access_key"}),
        }
    except Exception as e:
        print(f"Error processing task hierarchy: {e}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/progress/updateTask")
def receive_progress(progress: ProgressData):
    """Handles task progress updates (started/complete) and stores them."""
    agent_id = _authorize_agent_access(progress.agent_id, progress.access_key)

    if progress.status == "started" or progress.status == "pending":
        # Check for existing incomplete task to update
        for entry in progress_log:
            if (
                entry.get("agent_id") == agent_id
                and entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                entry.update(
                    {
                        "subtask_progress": [
                            subtask.model_dump()
                            for subtask in progress.subtask_progress
                        ],
                        "started_at": datetime.now(UTC),
                        "status": progress.status,
                    }
                )
                return {"message": "Progress updated", "data": entry}

        # New task entry
        new_entry = _progress_dump(progress, agent_id)
        new_entry["started_at"] = datetime.now(UTC)
        progress_log.append(new_entry)
        return {"message": "Progress received", "data": new_entry}

    elif progress.status == "complete":
        # Complete existing task
        for entry in progress_log:
            if (
                entry.get("agent_id") == agent_id
                and entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                entry.update(
                    {
                        "subtask_progress": [
                            subtask.model_dump()
                            for subtask in progress.subtask_progress
                        ],
                        "completed_at": datetime.now(UTC),
                        "completet_at": datetime.now(UTC),
                        "status": "complete",
                    }
                )
                return {"message": "Task completed", "data": entry}
        return {"message": f"No active task {progress.task_name} found."}

    else:
        raise HTTPException(400, "Status must be 'started', 'pending', or 'complete'.")


@router.get("/api/progress")
def get_progress_log(
    agent_id: str,
    access_key: Annotated[str | None, Header()] = None,
):
    authorized_agent_id = _authorize_agent_access(agent_id, access_key)
    return [
        entry
        for entry in progress_log
        if entry.get("agent_id") == authorized_agent_id
    ]
