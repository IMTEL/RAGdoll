"""Hash legacy plaintext agent access keys stored in MongoDB.

Access keys created before hashing are stored with their plaintext in
``access_key[].key``. For every such key that has no ``key_hash`` yet, this
script sets ``key_hash`` (SHA-256 of the key, as AccessService computes it)
and ``prefix`` (the first 6 characters, for display). The plaintext is left
in place so the change can be rolled back, until the script is run again with
``--drop-plaintext``, which removes ``key`` from every key whose stored hash
matches it.

Keys that were created with ``view_once: false`` (shown "anytime" before
hashing) also get ``key_encrypted``, the plaintext encrypted with FERNET_KEY,
so they can still be revealed via ``GET /agents/{id}/accesskeys/{key_id}/reveal``.
This happens before the plaintext is dropped, also when both steps run at once.
A key whose plaintext was already dropped cannot be made revealable again.

Idempotent: keys that are already hashed (or already dropped) are skipped, so
running it twice changes nothing the second time. Existing keys keep working
throughout: AccessService accepts both the hash and the legacy plaintext.

Each agent is written with a compare-and-set on its ``access_key`` array, so a
key created or revoked while the script runs is never overwritten; that agent
is reported as skipped and picked up by the next run.

Usage (from the repository root, with the backend's environment configured):

    uv run python scripts/migrate_hash_access_keys.py --dry-run
    uv run python scripts/migrate_hash_access_keys.py
    uv run python scripts/migrate_hash_access_keys.py --drop-plaintext --dry-run
    uv run python scripts/migrate_hash_access_keys.py --drop-plaintext
"""

import argparse
import copy
import hmac
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.access_service.access_service import display_prefix, hash_access_key
from src.utils.crypto_utils import encrypt_str


logger = logging.getLogger(__name__)


def _migrate_keys(
    keys: list[dict], drop_plaintext: bool
) -> tuple[list[dict], dict[str, int]]:
    """Returns the migrated key list and counts of keys hashed, encrypted, dropped."""
    migrated = copy.deepcopy(keys)
    counts = {"hashed": 0, "encrypted": 0, "dropped": 0}
    for entry in migrated:
        plaintext = entry.get("key")
        if not plaintext:
            continue
        if not entry.get("key_hash"):
            entry["key_hash"] = hash_access_key(plaintext)
            entry["prefix"] = display_prefix(plaintext)
            counts["hashed"] += 1
        # Legacy keys default to view-once: only an explicit False is revealable.
        if entry.get("view_once") is False and not entry.get("key_encrypted"):
            entry["key_encrypted"] = encrypt_str(plaintext)
            counts["encrypted"] += 1
        if drop_plaintext and hmac.compare_digest(
            entry["key_hash"], hash_access_key(plaintext)
        ):
            entry["key"] = None
            counts["dropped"] += 1
    return migrated, counts


def migrate(collection, dry_run: bool = False, drop_plaintext: bool = False) -> dict:
    """Migrates every agent in ``collection`` and returns counts of what changed."""
    counts = {
        "agents_updated": 0,
        "agents_skipped": 0,
        "hashed": 0,
        "encrypted": 0,
        "dropped": 0,
    }
    for doc in collection.find({}, {"access_key": 1}):
        keys = doc.get("access_key") or []
        migrated, key_counts = _migrate_keys(keys, drop_plaintext)
        if migrated == keys:
            continue
        if not dry_run:
            result = collection.update_one(
                {"_id": doc["_id"], "access_key": keys},
                {"$set": {"access_key": migrated}},
            )
            if result.modified_count == 0:
                logger.warning(
                    "Agent %s changed while migrating; run again", doc["_id"]
                )
                counts["agents_skipped"] += 1
                continue
        counts["agents_updated"] += 1
        for name, value in key_counts.items():
            counts[name] += value
    return counts


def _default_collection():
    from pymongo import MongoClient

    from src.config import Config

    config = Config()
    client = MongoClient(config.MONGODB_URI)
    return client[config.MONGODB_DATABASE][config.MONGODB_AGENT_COLLECTION]


def main(argv: list[str] | None = None, collection=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything",
    )
    parser.add_argument(
        "--drop-plaintext",
        action="store_true",
        help="Also remove the plaintext key once its hash is stored",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if collection is None:
        collection = _default_collection()
    counts = migrate(
        collection, dry_run=args.dry_run, drop_plaintext=args.drop_plaintext
    )
    logger.info(
        "%sagents updated: %s, keys hashed: %s, reusable keys encrypted: %s, "
        "plaintext keys dropped: %s, agents skipped (changed concurrently): %s",
        "[dry run] " if args.dry_run else "",
        counts["agents_updated"],
        counts["hashed"],
        counts["encrypted"],
        counts["dropped"],
        counts["agents_skipped"],
    )
    return 1 if counts["agents_skipped"] else 0


if __name__ == "__main__":
    sys.exit(main())
