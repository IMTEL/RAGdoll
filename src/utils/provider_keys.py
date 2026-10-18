"""Resolve agent provider keys that reference a user's stored API keys.

An agent can hold its provider keys literally (``llm_api_key``) or reference
one of its owner's stored keys by id (``llm_api_key_id``, a ``UserAPIKey.id``).
References are resolved here, server-side, at the moment a key is needed, so
the decrypted key never leaves the backend and deleting or replacing the
stored key takes effect immediately.
"""

import logging

from src.models.agent import Agent
from src.models.users.user import User
from src.rag_service.dao.user.base import UserDao
from src.utils.crypto_utils import decrypt_value


logger = logging.getLogger(__name__)

PROVIDER_KEY_FIELDS = ("llm_api_key", "embedding_api_key")
# Which UserAPIKey.usage values each agent key field accepts.
ALLOWED_USAGES = {
    "llm_api_key": {"llm", "both"},
    "embedding_api_key": {"embedding", "both"},
}


def find_agent_owner(agent_id: str | None, user_dao: UserDao) -> User | None:
    """Returns the user who owns ``agent_id``, or None."""
    if not agent_id:
        return None
    return next(
        (
            user
            for user in user_dao.get_users_with_agent(agent_id)
            if agent_id in user.owned_agents
        ),
        None,
    )


def resolve_provider_keys(agent: Agent, user_dao: UserDao) -> Agent:
    """Returns ``agent`` with referenced provider keys decrypted into place.

    Agents without references are returned unchanged. A reference whose stored
    key is gone (deleted, or the owner left) or cannot be decrypted resolves
    to an empty key with ``<field>_invalid`` set, exactly like a missing key:
    the LLM and embedding clients then refuse the call with a 401.
    """
    if not any(getattr(agent, f"{field}_id") for field in PROVIDER_KEY_FIELDS):
        return agent

    owner = find_agent_owner(agent.id, user_dao)
    stored_keys = {key.id: key for key in owner.api_keys} if owner else {}
    resolved = agent.model_copy()
    for field in PROVIDER_KEY_FIELDS:
        key_id = getattr(agent, f"{field}_id")
        if not key_id:
            continue
        setattr(resolved, field, "")
        stored = stored_keys.get(key_id)
        if stored is None:
            logger.warning(
                "Agent %s: stored API key referenced by %s_id no longer exists "
                "and must be selected again",
                agent.id,
                field,
            )
            setattr(resolved, f"{field}_invalid", True)
            continue
        try:
            setattr(resolved, field, decrypt_value(stored.key_encrypted))
        except ValueError:
            logger.warning(
                "Agent %s: stored API key referenced by %s_id cannot be "
                "decrypted with the configured FERNET_KEY",
                agent.id,
                field,
            )
            setattr(resolved, f"{field}_invalid", True)
    return resolved
