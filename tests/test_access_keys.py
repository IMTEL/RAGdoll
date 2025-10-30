from datetime import datetime

import pytest

from src.access_service.factory import AccessServiceConfig, access_service_factory
from src.models.agent import Agent, Role
from src.utils.crypto_utils import verify_access_key
from tests.mocks.mock_agent_dao import MockAgentDAO


def sample_agent() -> Agent:
    """Create a sample agent for testing."""
    return Agent(
        name="Test MongoDB Agent",
        description="A test agent for MongoDB DAO testing",
        prompt="You are a helpful test assistant. User info: {user_information}",
        corpa=["doc1", "doc2", "doc3"],
        roles=[
            Role(
                name="admin",
                description="Administrator with full access",
                subset_of_corpa=[0, 1, 2],
            ),
            Role(
                name="user",
                description="Regular user with limited access",
                subset_of_corpa=[1],
            ),
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
    )


# Example datetime for testing
DEFAULT_DATETIME = datetime(3000, 2, 23, 0, 0, 0)


def get_access_service_and_agent():
    database = MockAgentDAO()
    agent = sample_agent()
    database.add_agent(agent)
    access_service = access_service_factory(AccessServiceConfig("service", database))
    return agent, access_service, database


@pytest.mark.unit
def test_create_access_key():
    agent, access_service, database = get_access_service_and_agent()
    key = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)
    assert key.name == "test"
    assert key.expiry_date == DEFAULT_DATETIME

    stored_key = database.get_agent_by_id(agent.id).access_key[0]
    assert stored_key.id == key.id
    assert stored_key.name == key.name
    assert stored_key.expiry_date == key.expiry_date
    assert stored_key.key != key.key

    assert verify_access_key(key.key, stored_key.key.encode("utf-8"))


@pytest.mark.unit
def test_create_access_key_no_expiery():
    agent, access_service, database = get_access_service_and_agent()
    key = access_service.generate_accesskey("test", None, agent.id)
    assert key.name == "test"
    assert key.expiry_date is None

    stored_key = database.get_agent_by_id(agent.id).access_key[0]
    assert stored_key.id == key.id
    assert stored_key.key != key.key
    assert verify_access_key(key.key, stored_key.key.encode("utf-8"))

    assert access_service.authenticate(agent_id=agent.id, access_key=key.key)


@pytest.mark.unit
def test_revoke_access_key():
    agent, access_service, database = get_access_service_and_agent()
    returned_key = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)

    stored_key = database.get_agent_by_id(agent.id).access_key[0]
    assert verify_access_key(returned_key.key, stored_key.key.encode("utf-8"))

    assert access_service.revoke_key(agent.id, returned_key.id)
    assert len(database.get_agent_by_id(agent.id).access_key) == 0


@pytest.mark.unit
def test_bad_request():
    agent, access_service, _ = get_access_service_and_agent()
    with pytest.raises(ValueError):
        access_service.generate_accesskey(
            "test", datetime(2000, 2, 23, 0, 0, 0), agent.id
        )


@pytest.mark.unit
def test_authenticate_key():
    agent, access_service, database = get_access_service_and_agent()
    returned_key = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)

    assert access_service.authenticate(agent.id, returned_key.key)

    assert access_service.revoke_key(agent.id, returned_key.id)
    assert not access_service.authenticate(agent.id, returned_key.key)


@pytest.mark.unit
def test_authenticate_rejects_wrong_key():
    agent, access_service, _ = get_access_service_and_agent()
    returned_key = access_service.generate_accesskey("test", DEFAULT_DATETIME, agent.id)

    wrong = returned_key.key[:-2] + "AA"
    assert wrong != returned_key.key
    assert not access_service.authenticate(agent.id, wrong)
