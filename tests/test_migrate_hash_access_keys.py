"""Tests for scripts/migrate_hash_access_keys.py, run against mongomock."""

import copy
import hashlib
import importlib.util
from pathlib import Path

import mongomock
import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_hash_access_keys.py"
)
_spec = importlib.util.spec_from_file_location("migrate_hash_access_keys", _SCRIPT)
migration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migration)


LEGACY_A = "bGVnYWN5LWtleS1hLXZhbHVlLTMyLWJ5dGVzLWxvbmch"
LEGACY_B = "bGVnYWN5LWtleS1iLXZhbHVlLTMyLWJ5dGVzLWxvbmch"
NEW_KEY = "rgd_newkey01_" + "n" * 43


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def collection():
    coll = mongomock.MongoClient().db.agents
    coll.insert_many(
        [
            {
                "name": "legacy agent",
                "access_key": [
                    {"id": "legacy0a", "name": "a", "key": LEGACY_A, "view_once": True},
                    {"id": "legacy0b", "name": "b", "key": LEGACY_B},
                ],
            },
            {
                "name": "new agent",
                "access_key": [
                    {
                        "id": "newkey01",
                        "name": "new",
                        "key": None,
                        "key_hash": _sha256(NEW_KEY),
                        "prefix": NEW_KEY[:12],
                    }
                ],
            },
            {"name": "no keys"},
        ]
    )
    return coll


def _snapshot(coll) -> list[dict]:
    return copy.deepcopy(list(coll.find().sort("name")))


def _keys(coll, name: str) -> list[dict]:
    return coll.find_one({"name": name})["access_key"]


def test_dry_run_changes_nothing(collection):
    before = _snapshot(collection)
    assert migration.main(["--dry-run"], collection=collection) == 0
    assert _snapshot(collection) == before


def test_dry_run_with_drop_plaintext_changes_nothing(collection):
    migration.main([], collection=collection)
    before = _snapshot(collection)
    assert migration.main(["--dry-run", "--drop-plaintext"], collection=collection) == 0
    assert _snapshot(collection) == before


def test_adds_hash_and_prefix_and_keeps_plaintext(collection):
    assert migration.main([], collection=collection) == 0

    key_a, key_b = _keys(collection, "legacy agent")
    assert key_a["key_hash"] == _sha256(LEGACY_A)
    assert key_a["prefix"] == LEGACY_A[:6]
    assert key_a["key"] == LEGACY_A
    assert key_b["key_hash"] == _sha256(LEGACY_B)
    assert key_b["key"] == LEGACY_B


def test_leaves_new_format_keys_untouched(collection):
    before = _keys(collection, "new agent")
    migration.main([], collection=collection)
    assert _keys(collection, "new agent") == before


def test_is_idempotent(collection):
    migration.main([], collection=collection)
    after_first = _snapshot(collection)
    counts = migration.migrate(collection)
    assert counts["hashed"] == 0
    assert counts["agents_updated"] == 0
    assert _snapshot(collection) == after_first


def test_drop_plaintext_removes_key_after_hashing(collection):
    assert migration.main(["--drop-plaintext"], collection=collection) == 0

    for stored, legacy in zip(
        _keys(collection, "legacy agent"), (LEGACY_A, LEGACY_B), strict=True
    ):
        assert stored.get("key") is None
        assert stored["key_hash"] == _sha256(legacy)
        assert stored["prefix"] == legacy[:6]

    after = _snapshot(collection)
    migration.main(["--drop-plaintext"], collection=collection)
    assert _snapshot(collection) == after


def test_drop_plaintext_keeps_key_whose_hash_does_not_match(collection):
    collection.update_one(
        {"name": "legacy agent"},
        {"$set": {"access_key.0.key_hash": "0" * 64}},
    )
    migration.main(["--drop-plaintext"], collection=collection)
    key_a, key_b = _keys(collection, "legacy agent")
    assert key_a["key"] == LEGACY_A
    assert key_b.get("key") is None
