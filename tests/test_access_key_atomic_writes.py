"""Access-key writes are atomic.

Concurrent mints, revokes and agent updates never drop each other's changes.
"""

from datetime import UTC, datetime

import pytest

from src.access_service.access_service import AccessService
from src.models.accesskey import AccessKey
from src.rag_service.dao.agent.mongodb_agent_dao import MongoDBAgentDAO
from tests.mocks.mock_agent_dao import MockAgentDAO
from tests.test_access_keys import sample_agent


EXPIRY = datetime(3000, 1, 1, tzinfo=UTC)


@pytest.fixture(params=["mongodb", "mock"])
def dao(request):
    if request.param == "mongodb":
        yield MongoDBAgentDAO()
        return
    database = MockAgentDAO()
    database.clear()
    yield database
    database.clear()


@pytest.fixture
def agent_id(dao):
    return dao.add_agent(sample_agent()).id


def _key_names(dao, agent_id):
    return sorted(k.name for k in dao.get_agent_by_id(agent_id).access_key)


def _rename(dao, agent_id, name):
    agent = dao.get_agent_by_id(agent_id)
    agent.name = name
    dao.add_agent(agent)


def test_interleaved_mints_and_update_all_persist(dao, agent_id, monkeypatch):
    service = AccessService(dao)
    real_unique_id = service.get_unique_access_key_id
    state = {"interleaved": False}

    def unique_id_then_interleave(agent):
        # Mint "A" has read the agent; before it writes, another mint and an
        # owner's edit complete.
        if not state["interleaved"]:
            state["interleaved"] = True
            service.generate_accesskey("B", EXPIRY, agent_id)
            _rename(dao, agent_id, "Renamed by owner")
        return real_unique_id(agent)

    monkeypatch.setattr(service, "get_unique_access_key_id", unique_id_then_interleave)

    service.generate_accesskey("A", EXPIRY, agent_id)

    assert _key_names(dao, agent_id) == ["A", "B"]
    assert dao.get_agent_by_id(agent_id).name == "Renamed by owner"


def test_update_from_stale_read_keeps_minted_key(dao, agent_id):
    service = AccessService(dao)
    stale = dao.get_agent_by_id(agent_id)  # e.g. update_agent's existing_agent
    service.generate_accesskey("A", EXPIRY, agent_id)

    stale.name = "Edited"
    dao.add_agent(stale)

    assert _key_names(dao, agent_id) == ["A"]
    assert dao.get_agent_by_id(agent_id).name == "Edited"


def test_revoke_keeps_key_minted_meanwhile(dao, agent_id, monkeypatch):
    service = AccessService(dao)
    doomed = service.generate_accesskey("doomed", EXPIRY, agent_id)
    real_lookup = service.get_access_key_by_id

    def lookup_then_interleave(agent, key_id):
        found = real_lookup(agent, key_id)
        service.generate_accesskey("fresh", EXPIRY, agent_id)
        _rename(dao, agent_id, "Renamed during revoke")
        return found

    monkeypatch.setattr(service, "get_access_key_by_id", lookup_then_interleave)

    assert service.revoke_key(agent_id, doomed.id) is True

    assert _key_names(dao, agent_id) == ["fresh"]
    assert dao.get_agent_by_id(agent_id).name == "Renamed during revoke"


def test_dao_push_and_pull_access_keys(dao, agent_id):
    keys = [AccessKey(id=f"k{i}", name=f"n{i}", key_hash=str(i) * 64) for i in range(3)]
    for key in keys:
        assert dao.push_access_key(agent_id, key) is True

    assert dao.pull_access_keys(agent_id, ["k0", "k2", "missing"]) is True
    assert [k.id for k in dao.get_agent_by_id(agent_id).access_key] == ["k1"]
    assert dao.pull_access_keys(agent_id, ["missing"]) is False
