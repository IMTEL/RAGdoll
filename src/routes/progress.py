from fastapi import APIRouter, HTTPException
from typing import List
from datetime import datetime, timezone

from src.models.progress import ProgressData


# In-memory log to store progress data
from src.utils.global_logs import progressLog

# Define a router for progress-related endpoints
router = APIRouter()

@router.post("/api/progress")
def receive_progress(progress: ProgressData):
    """
    Receives progress data and stores it in memory.
    """
    # Starting tasks
    if progress.status == "start":
        progress.startedAt = datetime.now(timezone.utc)
        # Store the new progress data
        progressLog.append(progress.model_dump())
        return {"message": "Progress received successfully", "data": progress.model_dump()}

    # Completing tasks
    elif progress.status == "complete":
        # Look for an existing task with the same name that hasn't been completed yet
        for entry in progressLog:
            if entry["taskName"] == progress.taskName and entry["completedAt"] is None:
                # Update the existing task with completedAt and status
                entry["completedAt"] = datetime.now(timezone.utc)
                entry["status"] = "complete"  # Update status to "complete"
                print("Updated Progress Log: ", progressLog)  # Debugging
                # Return the updated entry instead of progress.model_dump()
                return {"message": "Progress received successfully", "data": entry}
        else:
            # Error if no task found
            return {"message": f"No started task found for {progress.taskName}. Cannot complete."}
    else: 
        # Error if status is neither "start" nor "complete"
        raise HTTPException(status_code=400, detail="Invalid status. Must be 'start' or 'complete'.")  # HTTP 400 Bad Request

