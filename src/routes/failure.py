from fastapi import APIRouter, HTTPException
from typing import List
from src.models.failure import failureData

failureLog = []

router = APIRouter()

@router.post("/api/failure")
def receive_failure(failure: failureData):
    """
    Receives failure data and stores it in memory.
    """
    # Store the failure data in the in-memory list
    failureLog.append(failure.model_dump())
    
    #print(failureLog) # Debugging
    
    return {"message": "Failure received successfully", "data": failure.model_dump()}

