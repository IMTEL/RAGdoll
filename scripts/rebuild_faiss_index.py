"""Rebuild FAISS vector indices from the MongoDB context collection.

Every context row stores its chunk's embedding, so an agent's FAISS index can
always be recreated from MongoDB: after a lost or corrupt index file, after the
dimension-mismatch error, or as a one-off migration.

Safe to run while the backend is up. The new files are swapped in atomically,
and a running backend notices the changed files and reloads them on its next
search or upload for that agent; it does not need a restart.

Usage (from the repository root, with the backend's environment configured):

    uv run python scripts/rebuild_faiss_index.py --agent-id <agent_id>
    uv run python scripts/rebuild_faiss_index.py --all
"""

import argparse
import logging
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo.errors import PyMongoError

from src.rag_service.dao.context.faiss_vector_store import (
    FAISSIndexError,
    FAISSVectorStoreManager,
)
from src.rag_service.dao.factory import get_context_dao


logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent-id", help="Rebuild the index of this agent")
    target.add_argument(
        "--all", action="store_true", help="Rebuild every agent with context rows"
    )
    args = parser.parse_args(argv)

    collection = get_context_dao().collection
    agent_ids = (
        sorted(a for a in collection.distinct("agent_id") if a)
        if args.all
        else [args.agent_id]
    )

    manager = FAISSVectorStoreManager()
    failed = 0
    for agent_id in agent_ids:
        try:
            count = manager.rebuild_index(agent_id, collection)
            print(f"{agent_id}: {count} vectors")
        except (FAISSIndexError, PyMongoError) as e:
            failed += 1
            print(f"{agent_id}: FAILED: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
