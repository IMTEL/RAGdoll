from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class StepProgressDTO(BaseModel):
    stepName: str
    repetitionNumber: int
    completed: bool

class SubtaskProgressDTO(BaseModel):
    subtaskName: str
    description: str
    completed: bool
    stepProgress: List[StepProgressDTO]

class ProgressData(BaseModel):
    taskName: str
    description: str
    status: str  # "start" or "complete"
    userId: Optional[str] = None
    subtaskProgress: List[SubtaskProgressDTO]
    startedAt: Optional[datetime] = None
    completedAt: Optional[datetime] = None