import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.fernet import Fernet

from src.access_service.access_service import (
    AccessKeyNotFoundError,
    AccessKeyNotRevealableError,
    AccessKeyUndecryptableError,
    AccessService,
    display_prefix,
    parse_access_key,
)
from src.models.accesskey import AccessKey, AccessKeyCreated, AccessKeyPublic
from src.models.agent import Agent, Role
from src.utils.crypto_utils import decrypt_value
from src.utils.path_dependencies import is_jwt_shaped
from tests.mocks.mock_agent_dao import MockAgentDAO


def sample_agent() -> Agent:
    """Create a sample agent for testing."""
    return Agent(
        name="Test MongoDB Agent",
        description="A test agent for MongoDB DAO testing",
        prompt="You are a helpful test assistant. User info: {user_information}",
        roles=[
            Role(name="admin", description="Administrator with full access"),
            Role(name="user", description="Regular user with limited access"),
        ],
        llm_model="gpt-4",
        llm_temperature=0.8,
        llm_max_tokens=2000,
        llm_api_key="test-mongodb-key",
        access_key=[],
        retrieval_method="semantic",
        embedding_model="text-embedding-ada-002",
        status="active",
        response_format="json",
        last_updated="2025-10-05T12:00:00Z",
        embedding_api_key="test-embedding-key",
    )


# Example datetime for testing
DEFAULT_DATETIME = datetime(3000, 2, 23, 0, 0, 0, tzinfo=UTC)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def env():
    database = MockAgentDAO()
    database.clear()
    agent = database.add_agent(sample_agent())
    yield agent, AccessService(database), database
    database.clear()


def _stored_keys(database: MockAgentDAO, agent_id: str) -> list[AccessKey]:
    return database.get_agent_by_id(agent_id).access_key


def _add_stored_key(database: MockAgentDAO, agent_id: str, key: AccessKey) -> None:
    database.push_access_key(agent_id, key)


class TestGenerate:
    def test_returns_plaintext_once_in_new_format(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)

        assert isinstance(created, AccessKeyCreated)
        assert created.name == "test"
        assert created.expiry_date == DEFAULT_DATETIME
        assert created.key.startswith(f"rgd_{created.id}_")
        assert len(created.id) == 8
        assert parse_access_key(created.key) == created.id
        assert created.prefix == created.key[:12]
        # Must not be mistaken for a user JWT by get_optional_user.
        assert not is_jwt_shaped(created.key)
        assert "." not in created.key

    def test_stores_only_hash_and_prefix(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey("test", None, agent.id)

        [stored] = _stored_keys(database, agent.id)
        assert stored.id == created.id
        assert stored.key is None
        assert stored.key_hash == _sha256(created.key)
        assert stored.prefix == created.key[:12]
        assert created.key not in stored.model_dump_json()

    def test_keys_are_unique(self, env):
        agent, access_service, _ = env
        first = access_service.generate_accesskey("a", None, agent.id)
        second = access_service.generate_accesskey("b", None, agent.id)
        assert first.key != second.key
        assert first.id != second.id

    def test_past_expiry_is_rejected(self, env):
        agent, access_service, _ = env
        with pytest.raises(ValueError):
            access_service.generate_accesskey(
                "test", datetime(2000, 2, 23, tzinfo=UTC), agent.id
            )

    def test_naive_expiry_is_treated_as_utc(self, env):
        agent, access_service, _ = env
        naive = datetime(3000, 2, 23, 0, 0, 0)
        created = access_service.generate_accesskey("test", naive, agent.id)
        assert created.expiry_date == naive.replace(tzinfo=UTC)


class TestViewOnce:
    def test_default_is_view_once_without_encrypted_copy(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey("test", None, agent.id)

        assert created.view_once is True
        assert created.revealable is False
        [stored] = _stored_keys(database, agent.id)
        assert stored.view_once is True
        assert stored.key_encrypted is None

    def test_reusable_key_stores_hash_and_encrypted_copy(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey(
            "test", None, agent.id, view_once=False
        )

        assert created.view_once is False
        assert created.revealable is True
        [stored] = _stored_keys(database, agent.id)
        assert stored.view_once is False
        assert stored.key is None
        assert stored.key_hash == _sha256(created.key)
        assert stored.key_encrypted
        assert created.key not in stored.model_dump_json()
        assert decrypt_value(stored.key_encrypted) == created.key

    def test_reusable_key_authenticates_by_hash(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey(
            "test", None, agent.id, view_once=False
        )
        assert access_service.authenticate(agent.id, created.key)
        assert not access_service.authenticate(agent.id, created.key[:-1] + "x")

    def test_public_view_never_has_secrets(self, env):
        agent, access_service, database = env
        access_service.generate_accesskey("test", None, agent.id, view_once=False)
        [stored] = _stored_keys(database, agent.id)

        public = AccessKeyPublic.model_validate(stored.model_dump())
        dumped = public.model_dump()
        assert dumped["revealable"] is True
        assert dumped["view_once"] is False
        for field in ("key", "key_hash", "key_encrypted"):
            assert field not in dumped
        assert stored.key_encrypted not in public.model_dump_json()


class TestReveal:
    def test_reveals_reusable_key(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey(
            "test", None, agent.id, view_once=False
        )
        assert access_service.reveal_key(agent.id, created.id) == created.key

    def test_view_once_key_is_not_revealable(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey("test", None, agent.id)
        with pytest.raises(AccessKeyNotRevealableError):
            access_service.reveal_key(agent.id, created.id)

    def test_unknown_key(self, env):
        agent, access_service, _ = env
        with pytest.raises(AccessKeyNotFoundError):
            access_service.reveal_key(agent.id, "missing0")

    def test_undecryptable_key(self, env, caplog):
        agent, access_service, database = env
        foreign = Fernet(Fernet.generate_key()).encrypt(b"rgd_gone0001_x").decode()
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="gone0001",
                name="rotated out",
                key_hash="0" * 64,
                key_encrypted=foreign,
                view_once=False,
            ),
        )
        with pytest.raises(AccessKeyUndecryptableError):
            access_service.reveal_key(agent.id, "gone0001")
        assert foreign not in caplog.text


class TestAuthenticate:
    def test_valid_key(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey("test", None, agent.id)
        assert access_service.authenticate(agent.id, created.key)

    def test_valid_key_updates_last_use(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey("test", None, agent.id)
        assert _stored_keys(database, agent.id)[0].last_use is None

        before = datetime.now(UTC)
        assert access_service.authenticate(agent.id, created.key)

        last_use = _stored_keys(database, agent.id)[0].last_use
        assert last_use is not None
        assert last_use >= before - timedelta(seconds=1)

    def test_wrong_secret_with_valid_key_id(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey("test", None, agent.id)
        forged = f"rgd_{created.id}_{'A' * 43}"
        assert not access_service.authenticate(agent.id, forged)

    def test_wrong_key(self, env):
        agent, access_service, _ = env
        access_service.generate_accesskey("test", None, agent.id)
        assert not access_service.authenticate(agent.id, "wrong-key")
        assert not access_service.authenticate(agent.id, "")

    def test_key_of_another_agent(self, env):
        agent, access_service, database = env
        other = database.add_agent(sample_agent())
        created = access_service.generate_accesskey("test", None, other.id)
        assert not access_service.authenticate(agent.id, created.key)

    def test_unknown_agent(self, env):
        agent, access_service, _ = env
        created = access_service.generate_accesskey("test", None, agent.id)
        assert not access_service.authenticate("999", created.key)

    def test_expired_key(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey(
            "test", datetime.now(UTC) + timedelta(hours=1), agent.id
        )
        key = database.get_agent_by_id(agent.id).access_key[0]
        key.expiry_date = datetime.now(UTC) - timedelta(seconds=5)
        # add_agent keeps the stored keys; replace the key itself.
        database.pull_access_keys(agent.id, [key.id])
        database.push_access_key(agent.id, key)

        assert not access_service.authenticate(agent.id, created.key)

    def test_key_expiring_soon_stored_as_naive_utc(self, env):
        # Mongo hands datetimes back as naive UTC. A key expiring in 30 minutes
        # must still work whatever the server's local timezone is.
        agent, access_service, database = env
        secret = "rgd_naiveutc_" + "s" * 43
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="naiveutc",
                name="soon",
                key_hash=_sha256(secret),
                prefix=secret[:12],
                expiry_date=(datetime.now(UTC) + timedelta(minutes=30)).replace(
                    tzinfo=None
                ),
                created=None,
                last_use=None,
            ),
        )
        assert access_service.authenticate(agent.id, secret)

    def test_key_expired_recently_stored_as_naive_utc(self, env):
        agent, access_service, database = env
        secret = "rgd_naivepst_" + "s" * 43
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="naivepst",
                name="just expired",
                key_hash=_sha256(secret),
                prefix=secret[:12],
                expiry_date=(datetime.now(UTC) - timedelta(minutes=30)).replace(
                    tzinfo=None
                ),
                created=None,
                last_use=None,
            ),
        )
        assert not access_service.authenticate(agent.id, secret)

    def test_legacy_plaintext_key(self, env):
        agent, access_service, database = env
        legacy = "bGVnYWN5LWtleS12YWx1ZS0zMi1ieXRlcy1sb25nISE="
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="legacy01",
                key=legacy,
                name="legacy",
                expiry_date=None,
                created=None,
                last_use=None,
            ),
        )
        assert access_service.authenticate(agent.id, legacy)
        assert not access_service.authenticate(agent.id, legacy[:-1] + "x")

    def test_migrated_legacy_key_without_plaintext(self, env):
        agent, access_service, database = env
        legacy = "bWlncmF0ZWQtbGVnYWN5LWtleS12YWx1ZS0zMiEhISE="
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="legacy02",
                key=None,
                key_hash=_sha256(legacy),
                prefix=legacy[:6],
                name="migrated",
                expiry_date=None,
                created=None,
                last_use=None,
            ),
        )
        assert access_service.authenticate(agent.id, legacy)

    def test_expired_legacy_key(self, env):
        agent, access_service, database = env
        legacy = "ZXhwaXJlZC1sZWdhY3kta2V5LXZhbHVlLTMyISEhISE="
        _add_stored_key(
            database,
            agent.id,
            AccessKey(
                id="legacy03",
                key=legacy,
                name="expired legacy",
                expiry_date=datetime.now(UTC) - timedelta(days=1),
                created=None,
                last_use=None,
            ),
        )
        assert not access_service.authenticate(agent.id, legacy)

    def test_revoked_key(self, env):
        agent, access_service, database = env
        created = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)
        assert access_service.authenticate(agent.id, created.key)

        assert access_service.revoke_key(agent.id, created.id)
        assert len(_stored_keys(database, agent.id)) == 0
        assert not access_service.authenticate(agent.id, created.key)


class TestDisplayPrefix:
    def test_new_format_shows_scheme_and_key_id(self):
        assert display_prefix("rgd_abcdef12_secretsecret") == "rgd_abcdef12"

    def test_legacy_key_shows_only_six_characters(self):
        assert display_prefix("bGVnYWN5LWtleS12YWx1ZQ==") == "bGVnYW"


class TestParseAccessKey:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("rgd_abcdef12_secret-with_underscores", "abcdef12"),
            ("rgd_abcdef12_", None),
            ("rgd__secret", None),
            ("bGVnYWN5LWtleQ==", None),
            ("", None),
        ],
    )
    def test_parse(self, value, expected):
        assert parse_access_key(value) == expected
