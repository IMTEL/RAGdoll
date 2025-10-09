"""Training progress tracking domain models."""

from datetime import datetime

from pydantic import BaseModel, Field


class StepProgressDTO(BaseModel):
    """Progress tracking for a single step within a subtask.

    Attributes:
        step_name: Identifier for the step
        repetition_number: Which iteration/repetition of this step
        completed: Whether this step is finished
    """

    step_name: str
    repetition_number: int = Field(default=0, ge=0)
    completed: bool = Field(default=False)


class SubtaskProgressDTO(BaseModel):
    """Progress tracking for a subtask within a main task.

    Attributes:
        subtask_name: Identifier for the subtask
        description: Explanation of what this subtask involves
        completed: Whether this subtask is finished
        step_progress: List of steps within this subtask
    """

    subtask_name: str
    description: str
    completed: bool = Field(default=False)
    step_progress: list[StepProgressDTO] = Field(default_factory=list)


class ProgressData(BaseModel):
    """Complete progress tracking for a training task.

    Tracks the user's progress through a structured training program
    with tasks, subtasks, and steps. Used for educational/training
    scenarios in VR environments.

    Attributes:
        task_name: Identifier for the main task
        description: What this task teaches/covers
        status: Current state ("started", "complete")
        user_id: Optional identifier for the user
        subtask_progress: List of subtasks and their progress
        started_at: When the task was begun
        completed_at: When the task was finished (typo: should be completed_at)
    """

    task_name: str
    description: str
    status: str = Field(default="started")
    user_id: str | None = None
    subtask_progress: list[SubtaskProgressDTO] = Field(default_factory=list)
    started_at: datetime | None = None
    completet_at: datetime | None = (
        None  # Note: typo in original, keeping for compatibility
    )


class ListProgressData(BaseModel):
    """Container for multiple progress data items.

    Attributes:
        items: List of progress data entries
    """

    items: list[ProgressData] = Field(default_factory=list)
