import hashlib
import hmac
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from src.access_service.base import AbstractAccessService
from src.models.accesskey import (
    AccessKey,
    AccessKeyCreated,
    AccessKeyPublic,
    as_utc,
)
from src.models.agent import Agent
from src.rag_service.dao.agent.base import AgentDAO


logger = logging.getLogger(__name__)

ACCESS_KEY_SCHEME = "rgd"
ACCESS_KEY_PREFIX_LENGTH = 12
# Legacy keys have no public id part, so their prefix is pure secret: show less.
LEGACY_PREFIX_LENGTH = 6
# last_use is a display hint; don't write the agent on every single call.
LAST_USE_RESOLUTION = timedelta(minutes=1)


def hash_access_key(access_key: str) -> str:
    """SHA-256 hex digest of the full access key.

    Keys are 256-bit random values, so a fast hash is enough: there is nothing
    to brute-force, and lookups stay cheap.
    """
    return hashlib.sha256(access_key.encode("utf-8")).hexdigest()


def parse_access_key(access_key: str) -> str | None:
    """Returns the key id of an ``rgd_<keyid>_<secret>`` key, or None.

    Legacy keys (plain base64) have no key id and return None.
    """
    scheme, sep, rest = access_key.partition("_")
    if scheme != ACCESS_KEY_SCHEME or not sep:
        return None
    key_id, sep, secret = rest.partition("_")
    if not key_id or not sep or not secret:
        return None
    return key_id


def display_prefix(access_key: str) -> str:
    """The part of a key that is stored and shown so users can tell keys apart.

    ``rgd_<keyid>`` for new keys (the id is not secret); only the first 6
    characters for legacy keys, whose every character is secret.
    """
    if parse_access_key(access_key) is not None:
        return access_key[:ACCESS_KEY_PREFIX_LENGTH]
    return access_key[:LEGACY_PREFIX_LENGTH]


def _matches(stored: AccessKey, access_key: str, key_hash: str) -> bool:
    if stored.key_hash and hmac.compare_digest(stored.key_hash, key_hash):
        return True
    # Legacy plaintext key, until scripts/migrate_hash_access_keys.py drops it.
    return bool(stored.key) and hmac.compare_digest(
        stored.key.encode("utf-8"), access_key.encode("utf-8")
    )


class AccessService(AbstractAccessService):
    def __init__(self, agent_database: AgentDAO):
        self.agent_database = agent_database

    def try_get_agent(self, agent_id: str):
        agent = self.agent_database.get_agent_by_id(agent_id)
        if agent is None:
            logger.warning(f"Agent no found : {agent_id}")
            raise ValueError(f"No agent found by id : {agent_id}")
        return agent

    def get_unique_access_key_id(self, agent: Agent):
        tries = 0
        existing_keys = {ak.id for ak in agent.access_key}
        while tries < 50:
            new_id = uuid.uuid4().hex[:8]  # 8 Byte should be enough
            if new_id not in existing_keys:
                return new_id
            tries += 1
        raise ValueError("Could not generate a new key!")

    def generate_accesskey(
        self,
        name: str,
        expiry_date: datetime | None,
        agent_id: str,
    ) -> AccessKeyCreated:
        expiry_date = as_utc(expiry_date)
        now = datetime.now(UTC)
        if expiry_date is not None and expiry_date < now:
            raise ValueError("Expiry date cannot be in the past")

        agent = self.try_get_agent(agent_id)
        key_id = self.get_unique_access_key_id(agent)
        plaintext = f"{ACCESS_KEY_SCHEME}_{key_id}_{secrets.token_urlsafe(32)}"
        stored = AccessKey(
            id=key_id,
            name=name,
            key_hash=hash_access_key(plaintext),
            prefix=display_prefix(plaintext),
            expiry_date=expiry_date,
            created=now,
            last_use=None,
        )
        # $push, not a rewrite of the agent read above: a concurrent mint
        # or agent update must not be overwritten by this stale copy.
        if not self.agent_database.push_access_key(agent.id, stored):
            raise ValueError(f"Agent {agent_id} not found")
        return AccessKeyCreated(
            **stored.model_dump(include=set(AccessKeyPublic.model_fields)),
            key=plaintext,
        )

    def authenticate(self, agent_id: str, access_key: str) -> bool:
        if not access_key:
            return False
        try:
            agent = self.try_get_agent(agent_id)
        except Exception as e:
            logger.warning(f"An exception occured while trying to fetch agent {e}")
            return False

        key_id = parse_access_key(access_key)
        if key_id is not None:
            candidates = [ak for ak in agent.access_key if ak.id == key_id]
        else:
            candidates = agent.access_key
        key_hash = hash_access_key(access_key)
        match = next(
            (ak for ak in candidates if _matches(ak, access_key, key_hash)), None
        )
        if match is None:
            logger.warning("No matching access key found in agent")
            return False

        now = datetime.now(UTC)
        expiry_date = as_utc(match.expiry_date)
        if expiry_date is not None and expiry_date <= now:
            logger.warning(f"Access key with name : {match.name} has expired")
            return False

        last_use = as_utc(match.last_use)
        if last_use is None or now - last_use >= LAST_USE_RESOLUTION:
            try:
                self.agent_database.touch_access_key(agent.id, match.id, now)
            # last_use is informational: a failed write must not deny access.
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not update last_use of access key: {e}")
        return True

    def get_access_key_by_id(
        self, agent: Agent, access_key_id: str
    ) -> AccessKey | None:
        for ak in agent.access_key:
            if ak.id == access_key_id:
                return ak
        return None

    def revoke_key(self, agent_id: str, access_key_id: str) -> bool:
        agent = self.try_get_agent(agent_id)
        key = self.get_access_key_by_id(agent, access_key_id)
        if key is None:
            return False
        self.agent_database.pull_access_keys(agent.id, [key.id])
        return True

    def revoke_keys(self, agent_id: str, access_key_ids: list[str]) -> bool:
        """Remove several keys of the agent in one atomic update."""
        return self.agent_database.pull_access_keys(agent_id, access_key_ids)
