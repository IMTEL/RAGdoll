# Replace the real Mongo driver with an in-memory one before any ``src`` module
# is imported: DAOs are built at import time (src/globals.py) and bind
# ``MongoClient`` by name, so this must run first.
#
# USE_MONGOMOCK=false (environment or .env) keeps the real pymongo instead;
# the app then connects to MOCK_MONGODB_URI / MOCK_MONGODB_DATABASE. Use it to
# check behaviour mongomock does not model (TTL indexes, explain, ...).
import mimetypes
import os
import sys
import types

import mongomock
import pymongo
from dotenv import load_dotenv


load_dotenv()
USE_MONGOMOCK = os.getenv("USE_MONGOMOCK", "true").strip().lower() != "false"
if USE_MONGOMOCK:
    pymongo.MongoClient = mongomock.MongoClient

if sys.platform == "win32":
    # Windows: loading the libmagic DLL crashes the interpreter (access violation)
    # when ``unstructured`` imports ``magic``; CI on Linux uses the real library.
    # Provide a minimal stand-in that guesses the MIME type from the extension.
    def _guess_mime(path: str | None = None, data: bytes = b"") -> str:
        if data.startswith(b"%PDF"):
            return "application/pdf"
        guessed = mimetypes.guess_type(path or "")[0]
        return guessed or "application/octet-stream"

    class _FakeMagic:
        def __init__(self, mime: bool = False, **_kwargs):
            self.mime = mime

        def from_file(self, path):
            return _guess_mime(str(path))

        def from_buffer(self, data):
            return _guess_mime(None, bytes(data[:16]))

    _fake_magic = types.ModuleType("magic")
    _fake_magic.Magic = _FakeMagic
    _fake_magic.from_file = lambda path, mime=False: _guess_mime(str(path))
    _fake_magic.from_buffer = lambda data, mime=False: _guess_mime(
        None, bytes(data[:16])
    )
    sys.modules["magic"] = _fake_magic

import pytest

from src.config import Config


def pytest_configure(config):
    # Initialize configuration before tests run
    backend_config = Config()  # This ensures .env is loaded

    # Check if RUNNING_TESTS is set to True
    if not backend_config.RUNNING_TESTS:
        pytest.exit("RUNNING_TESTS is not set to True. Exiting tests.")


@pytest.fixture(autouse=True)
def _isolated_faiss_index_dir(tmp_path_factory, monkeypatch):
    # Never read or write the real /data/faiss_indices from tests, and start
    # every test with an empty FAISS store cache.
    from src.rag_service.dao.context import faiss_vector_store

    monkeypatch.setattr(
        faiss_vector_store,
        "FAISS_INDEX_DIR",
        tmp_path_factory.mktemp("faiss_indices"),
    )
    monkeypatch.setattr(faiss_vector_store.FAISSVectorStoreManager, "_stores", {})


@pytest.fixture
def progress_store(monkeypatch):
    """A fresh in-memory progress DAO, installed in every route module."""
    from src.rag_service.dao.progress.mongodb_progress_dao import MongoDBProgressDAO
    from src.routes import debug, progress, upload

    dao = MongoDBProgressDAO()
    for module in (progress, upload, debug):
        monkeypatch.setattr(module, "progress_dao", dao)
    return dao
