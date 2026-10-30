"""Shared client key for The Candidate (the VR app).

The key only lets a client list published agents (GET /agents/published) and
chat with them and with the observer agent set in CANDIDATE_OBSERVER_AGENT_ID
(the chat routes). It is deliberately not accepted by
AccessService.authenticate, so it never unlocks routes that return an agent's
configuration, such as /agents/{agent_id} or /agents/agent-info.
"""

import secrets

from src.config import Config
from src.models.agent import Agent


def is_candidate_client_key(credentials: str | None) -> bool:
    """True if ``credentials`` is the configured Candidate client key.

    Always False when CANDIDATE_CLIENT_KEY is empty, which turns the feature off.
    """
    expected = Config().CANDIDATE_CLIENT_KEY
    if not expected or not credentials:
        return False
    return secrets.compare_digest(credentials.strip().encode(), expected.encode())


def candidate_may_chat_with(agent: Agent) -> bool:
    """True if The Candidate's client key may chat with ``agent``.

    That is a published agent, or the observer that gives feedback after an
    interview. The observer is not published, so it is never listed.
    """
    observer_id = Config().CANDIDATE_OBSERVER_AGENT_ID
    return agent.is_published or (bool(observer_id) and agent.id == observer_id)
