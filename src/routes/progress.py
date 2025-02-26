from fastapi import APIRouter, HTTPException
from typing import List

from src.models.progress import ProgressData

# In-memory log to store progress data
progressLog = []

# Define a router for progress-related endpoints
router = APIRouter()

@router.post("/api/progress")
def receive_progress(progress: ProgressData):
    """
    Receives progress data and stores it in memory.
    """
    # Store the progress data in the in-memory list
    progressLog.append(progress.model_dump_json())
    
    print(progressLog)
    
    return {"message": "Progress received successfully", "data": progress.dict()}
    