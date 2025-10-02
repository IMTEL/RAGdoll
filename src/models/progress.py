from datetime import datetime

from pydantic import BaseModel


class StepProgressDTO(BaseModel):
    step_name: str
    repetition_number: int
    completed: bool


class SubtaskProgressDTO(BaseModel):
    subtask_name: str
    description: str
    completed: bool
    step_progress: list[StepProgressDTO]


class ProgressData(BaseModel):
    task_name: str
    description: str
    status: str  # "started" or "complete"
    user_id: str | None = None
    subtask_progress: list[SubtaskProgressDTO]
    started_at: datetime | None = None
    completet_at: datetime | None = None


class ListProgressData(BaseModel):
    items: list[ProgressData]
