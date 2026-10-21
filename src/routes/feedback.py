"""Endpoints for counsellor feedback on agent replies.

Anyone who can access an agent (its owner or a collaborator) can rate its
replies, list the feedback and delete it. An agent the user cannot access
is reported as 404, so its existence is not revealed. The most recent
feedback is added to the agent's prompt by the chat pipeline.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from src.globals import feedback_dao
from src.models.feedback import MessageFeedback, Rating
from src.models.users.user import User
from src.rag_service.dao.feedback.base import FeedbackDAO
from src.utils.path_dependencies import require_user, require_user_has_agent_access


router = APIRouter(prefix="/agents/{agent_id}/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Body for rating an agent reply.

    ``agent_id`` comes from the path and ``author_id`` from the logged-in
    user. Unknown fields are rejected.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    rating: Rating
    agent_message: str = Field(..., min_length=1, max_length=4000)
    user_message: str = Field(default="", max_length=4000)
    comment: str = Field(default="", max_length=500)


def get_dao() -> FeedbackDAO:
    """Return the feedback DAO."""
    return feedback_dao


def get_current_user(user: Annotated[User, Depends(require_user)]) -> User:
    """Return the authenticated user."""
    return user


def get_agent_user(
    agent_id: str, user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Return the user if they can access the agent, or raise 404."""
    require_user_has_agent_access(agent_id, user)
    return user


AgentUser = Annotated[User, Depends(get_agent_user)]
Dao = Annotated[FeedbackDAO, Depends(get_dao)]


@router.get("", response_model=list[MessageFeedback])
def list_feedback(
    agent_id: str,
    _user: AgentUser,
    dao: Dao,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    """List the agent's feedback, newest first."""
    return dao.get_recent_by_agent(agent_id, limit)


@router.post("", response_model=MessageFeedback, status_code=status.HTTP_201_CREATED)
def create_feedback(agent_id: str, payload: FeedbackRequest, user: AgentUser, dao: Dao):
    """Rate one of the agent's replies."""
    feedback = MessageFeedback(
        agent_id=agent_id, author_id=user.id or "", **payload.model_dump()
    )
    return dao.create(feedback)


@router.delete("/{feedback_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_feedback(agent_id: str, feedback_id: str, _user: AgentUser, dao: Dao):
    """Delete a feedback entry of the agent, e.g. to undo a rating."""
    feedback = dao.get_by_id(feedback_id)
    if feedback is None or feedback.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Feedback not found")
    if not dao.delete(feedback_id):
        raise HTTPException(status_code=404, detail="Feedback not found")
