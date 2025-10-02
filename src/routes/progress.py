from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from src.models.progress import ListProgressData, ProgressData

# In-memory log to store progress data
from src.utils.global_logs import progress_log


# Define a router for progress-related endpoints
router = APIRouter()


@router.post("/api/progress/initializeTasks")
def receive_hierarchy(task_hierarchy: ListProgressData):
    """Initializes a list of tasks with their subtasks and steps."""
    try:
        for task in task_hierarchy.items:
            progress_log.append(task.model_dump())
        return {"message": "Tasks initialized", "data": task_hierarchy}
    except Exception as e:
        print(f"Error processing task hierarchy: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/progress/updateTask")
def receive_progress(progress: ProgressData):
    """Handles task progress updates (started/complete) and stores them."""
    if progress.status == "started" or progress.status == "pending":
        # Check for existing incomplete task to update
        for entry in progress_log:
            if (
                entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                entry.update(
                    {
                        "subtask_progress": progress.subtask_progress,
                        "started_at": datetime.now(UTC),
                    }
                )
                return {"message": "Progress updated", "data": entry}

        # New task entry
        new_entry = progress.model_dump()
        new_entry["started_at"] = datetime.now(UTC)
        progress_log.append(new_entry)
        return {"message": "Progress received", "data": new_entry}

    elif progress.status == "complete":
        # Complete existing task
        for entry in progress_log:
            if (
                entry["task_name"] == progress.task_name
                and entry["completed_at"] is None
            ):
                entry.update(
                    {
                        "subtask_progress": progress.subtask_progress,
                        "completed_at": datetime.now(UTC),
                        "status": "complete",
                    }
                )
                return {"message": "Task completed", "data": entry}
        return {"message": f"No active task {progress.task_name} found."}

    else:
        raise HTTPException(400, "Status must be 'started' or 'complete'.")


@router.get("/api/progress")
def get_progress_log():
    return progress_log  # Returns the entire in-memory list
