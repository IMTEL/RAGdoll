"""Tests for MongoDBFeedbackDAO.

They run against mongomock by default (see tests/conftest.py) or a real
MongoDB with USE_MONGOMOCK=false. Each test uses its own agent ids, so it
never touches other data in the collection.
"""

import uuid
from datetime import datetime, timedelta

import pytest

from src.models.feedback import MessageFeedback
from src.rag_service.dao.feedback.mongodb_feedback_dao import MongoDBFeedbackDAO


@pytest.fixture
def repo() -> MongoDBFeedbackDAO:
    repo = MongoDBFeedbackDAO()
    if not repo.is_reachable():
        pytest.skip("MongoDB is not reachable.")
    return repo


@pytest.fixture
def agent_id(repo: MongoDBFeedbackDAO):
    agent_id = f"agent-{uuid.uuid4()}"
    yield agent_id
    repo.delete_by_agent(agent_id)


def _feedback(agent_id: str, **fields) -> MessageFeedback:
    body = {
        "agent_id": agent_id,
        "author_id": "alice",
        "rating": "good",
        "agent_message": "Fortell litt om deg selv.",
        **fields,
    }
    return MessageFeedback(**body)


def _insert_at(repo: MongoDBFeedbackDAO, agent_id: str, minutes_ago: int) -> str:
    """Insert feedback with a fixed timestamp, bypassing ``create``."""
    feedback_id = str(uuid.uuid4())
    doc = _feedback(agent_id, agent_message=f"reply {minutes_ago}").model_dump(
        exclude={"id"}
    )
    doc["_id"] = feedback_id
    doc["created_at"] = datetime.now() - timedelta(minutes=minutes_ago)
    repo.collection.insert_one(doc)
    return feedback_id


class TestMongoDBFeedbackDAO:
    def test_create_sets_id_and_timestamp(self, repo, agent_id):
        created = repo.create(_feedback(agent_id, comment="Bra spørsmål"))

        assert created.id
        stored = repo.get_by_id(created.id)
        assert stored is not None
        assert stored.agent_id == agent_id
        assert stored.rating == "good"
        assert stored.comment == "Bra spørsmål"

    def test_create_requires_author(self, repo, agent_id):
        with pytest.raises(ValueError):
            repo.create(_feedback(agent_id, author_id=""))

    def test_get_by_id_unknown(self, repo):
        assert repo.get_by_id("missing") is None
        assert repo.get_by_id("") is None

    def test_recent_is_newest_first_and_limited(self, repo, agent_id):
        for minutes_ago in (30, 10, 20, 40):
            _insert_at(repo, agent_id, minutes_ago)

        recent = repo.get_recent_by_agent(agent_id, 3)

        assert [f.agent_message for f in recent] == [
            "reply 10",
            "reply 20",
            "reply 30",
        ]

    def test_recent_only_returns_the_agent(self, repo, agent_id):
        other = f"agent-{uuid.uuid4()}"
        repo.create(_feedback(agent_id))
        repo.create(_feedback(other))
        try:
            recent = repo.get_recent_by_agent(agent_id, 10)
            assert [f.agent_id for f in recent] == [agent_id]
        finally:
            repo.delete_by_agent(other)

    def test_recent_with_no_limit_is_empty(self, repo, agent_id):
        repo.create(_feedback(agent_id))
        assert repo.get_recent_by_agent(agent_id, 0) == []

    def test_delete(self, repo, agent_id):
        created = repo.create(_feedback(agent_id))

        assert repo.delete(created.id) is True
        assert repo.get_by_id(created.id) is None
        assert repo.delete(created.id) is False

    def test_delete_by_agent(self, repo, agent_id):
        other = f"agent-{uuid.uuid4()}"
        repo.create(_feedback(agent_id))
        repo.create(_feedback(agent_id))
        kept = repo.create(_feedback(other))
        try:
            assert repo.delete_by_agent(agent_id) == 2
            assert repo.get_recent_by_agent(agent_id, 10) == []
            assert repo.get_by_id(kept.id) is not None
        finally:
            repo.delete_by_agent(other)
