"""Counsellor feedback is added to the agent's prompt on every turn."""

from unittest.mock import MagicMock

import pytest

from src import pipeline
from src.models.chat.command import Command
from src.models.chat.message import Message
from src.models.feedback import MessageFeedback, Rating
from tests.endpoints.conftest import sample_agent


def _feedback(rating: Rating = "good", agent_message: str = "reply", comment: str = ""):
    return MessageFeedback(
        agent_id="a1",
        author_id="alice",
        rating=rating,
        agent_message=agent_message,
        comment=comment,
    )


@pytest.fixture
def dao(monkeypatch):
    dao = MagicMock()
    dao.get_recent_by_agent.return_value = []
    monkeypatch.setattr(pipeline, "feedback_dao", dao)
    return dao


class TestFeedbackPromptSection:
    def test_empty_without_feedback(self, dao):
        assert pipeline.feedback_prompt_section("a1") == ""

    def test_empty_without_agent_id(self, dao):
        assert pipeline.feedback_prompt_section("") == ""
        dao.get_recent_by_agent.assert_not_called()

    def test_asks_for_the_most_recent_entries(self, dao):
        pipeline.feedback_prompt_section("a1")

        dao.get_recent_by_agent.assert_called_once_with(
            "a1", pipeline.FEEDBACK_PROMPT_LIMIT
        )
        assert pipeline.FEEDBACK_PROMPT_LIMIT == 10

    def test_lists_entries_with_labels_in_dao_order(self, dao):
        dao.get_recent_by_agent.return_value = [
            _feedback("bad", "Too long answer", "Ask one question at a time"),
            _feedback("good", "Short answer"),
        ]

        section = pipeline.feedback_prompt_section("a1")

        assert section.startswith(pipeline.FEEDBACK_RULES)
        assert section.splitlines()[-2:] == [
            (
                '- BAD | Your reply: "Too long answer" '
                '| Comment: "Ask one question at a time"'
            ),
            '- GOOD | Your reply: "Short answer"',
        ]

    def test_explains_the_labels_and_priority(self):
        rules = pipeline.FEEDBACK_RULES
        assert "GOOD" in rules
        assert "BAD" in rules
        assert "Never mention it" in rules
        assert "those instructions win" in rules

    def test_long_texts_are_shortened_to_one_line(self, dao):
        dao.get_recent_by_agent.return_value = [
            _feedback("bad", "line one\n\nline two " + "x" * 500, "c" * 500)
        ]

        line = pipeline.feedback_prompt_section("a1").splitlines()[-1]

        assert "line one line two" in line
        reply = line.split('Your reply: "')[1].split('"')[0]
        comment = line.split('Comment: "')[1].rstrip('"')
        assert len(reply) == pipeline.FEEDBACK_TRUNCATE_LENGTH
        assert reply.endswith("...")
        assert len(comment) == pipeline.FEEDBACK_TRUNCATE_LENGTH

    def test_lookup_error_does_not_break_the_chat(self, dao):
        dao.get_recent_by_agent.side_effect = RuntimeError("mongo down")

        assert pipeline.feedback_prompt_section("a1") == ""


class TestFeedbackInMessages:
    def _build(self, **overrides):
        kwargs = {
            "persona": "PERSONA-TEXT",
            "role_prompt": None,
            "function_prompt": "",
            "retrieved_contexts": [],
            "retrieval_status": "empty",
            "game_context": "",
            "chat_log": [Message(role="user", content="hello")],
        }
        kwargs.update(overrides)
        return pipeline.build_messages(**kwargs)

    def test_feedback_follows_character_rules(self):
        rules = self._build(feedback="FEEDBACK-TEXT")[0]["content"]

        assert rules.index(pipeline.CHARACTER_RULES) < rules.index("FEEDBACK-TEXT")
        assert rules.index("PERSONA-TEXT") < rules.index("FEEDBACK-TEXT")

    def test_no_feedback_by_default(self):
        assert self._build() == self._build(feedback="")

    def test_pipeline_sends_feedback_to_llm(self, dao, monkeypatch):
        dao.get_recent_by_agent.return_value = [_feedback("bad", "FEEDBACK-REPLY")]
        agent = sample_agent()
        agent.id = "a1"
        agent.context_aware_retrieval = False
        fake_llm = MagicMock()
        fake_llm.generate.return_value = "answer"
        monkeypatch.setattr(pipeline, "create_llm", lambda *a, **k: fake_llm)
        context_dao = MagicMock()
        context_dao.get_context_for_agent.return_value = []
        monkeypatch.setattr(pipeline, "get_context_dao", lambda: context_dao)
        emb = MagicMock()
        emb.get_embedding.return_value = [0.0]
        monkeypatch.setattr(pipeline, "create_embeddings_model", lambda *a, **k: emb)

        pipeline.assemble_prompt_with_agent(
            Command(agent_id="a1", chat_log=[Message(role="user", content="q")]),
            agent,
        )

        messages = fake_llm.generate.call_args.args[0]
        assert '- BAD | Your reply: "FEEDBACK-REPLY"' in messages[0]["content"]
        dao.get_recent_by_agent.assert_called_once_with("a1", 10)
