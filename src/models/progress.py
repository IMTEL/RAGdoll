from datetime import datetime

from pydantic import BaseModel


class StepProgressDTO(BaseModel):
    stepName: str
    repetitionNumber: int
    completed: bool


class SubtaskProgressDTO(BaseModel):
    subtaskName: str
    description: str
    completed: bool
    stepProgress: list[StepProgressDTO]


class ProgressData(BaseModel):
    taskName: str
    description: str
    status: str  # "started" or "complete"
    userId: str | None = None
    subtaskProgress: list[SubtaskProgressDTO]
    startedAt: datetime | None = None
    completedAt: datetime | None = None


class ListProgressData(BaseModel):
    items: list[ProgressData]
