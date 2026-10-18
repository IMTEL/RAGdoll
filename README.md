# RAGdoll

FastAPI backend for a VR chatbot with Retrieval-Augmented Generation (RAG). Hosted instance: https://iplvr.it.ntnu.no

## Prerequisites
- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/) (Python 3.12)
- [Docker](https://www.docker.com/products/docker-desktop) for MongoDB, Keycloak and Mailpit

## Setup
```sh
git clone https://github.com/IMTEL/RAGdoll.git
cd RAGdoll
uv sync --dev
cp .env.example .env   # then edit the values
```

`.env.example` is the source of truth for configuration. Notable settings:

| Variable | Purpose |
|---|---|
| `FERNET_KEY` | Encrypts stored provider API keys. Accepts a comma-separated list: the first key encrypts, all keys decrypt (rotation). |
| `KEYCLOAK_*` | Keycloak realm, client and issuer/JWKS settings for login. |
| `SMTP_*` | Outgoing mail for Keycloak (password reset). Unset = local Mailpit. |
| `AUTH_BOOTSTRAP_ENABLED` | Debug bootstrap of auth data (needs a logged-in user). Keep `false` in production. |
| `PUBLIC_SPEECH_ENDPOINTS` | Default `false`: STT, TTS and translate need `Authorization: Bearer <user JWT or access key>`; with an access key also send `X-Agent-Id` (not needed with a user JWT). |
| `LOG_LEVEL`, `LOG_FILE` | Log level (default `INFO`) and optional rotating log file. |
| `LOG_CONVERSATIONS` | Default `false`. When `true`, user messages and agent responses are logged. |
| `LLM_TIMEOUT_SECONDS` | Timeout for LLM and embedding requests. |
| `STT_MAX_CONCURRENCY`, `TTS_MAX_CONCURRENCY` | Concurrent Whisper / Piper jobs. |
| `MAX_UPLOAD_MB` | Maximum document upload size (default 50). |
| `PROGRESS_TTL_HOURS` | Lifetime of upload progress records (default 24). |
| `MONGODB_PROGRESS_COLLECTION`, `MONGODB_UPLOAD_TASKS_COLLECTION` | Collections for progress and upload tasks. |

## Running locally

### With Docker Compose (backend, MongoDB, Keycloak, Mailpit)
```sh
docker compose -f docker-compose.local.yml up --build
```
- API: http://localhost:8000 (docs at `/docs`)
- Keycloak: http://localhost:8080
- Mailpit inbox (password reset emails): http://localhost:8025

### Backend only
Start MongoDB (and Keycloak if you need login) separately, then:
```sh
uv run uvicorn src.main:app --reload
curl http://127.0.0.1:8000/ping    # {"status":"I AM ALIVE!"}
```

## Tests
```sh
uv run pytest
```
`pytest-env` sets `RUNNING_TESTS=true`, which makes the app use test-safe configuration. No MongoDB is needed: `tests/conftest.py` replaces pymongo with mongomock (this is also what CI runs).

To run against a real MongoDB instead, set `USE_MONGOMOCK=false` (environment or `.env`; default `true`). The tests then use `MOCK_MONGODB_URI` and `MOCK_MONGODB_DATABASE`, which is how to check behaviour mongomock does not model, such as TTL indexes:
```sh
USE_MONGOMOCK=false MOCK_MONGODB_URI=mongodb://localhost:27017 uv run pytest
```

Lint and format checks:
```sh
uv run ruff check --no-fix src tests
uv run ruff format --check src tests
```

## Upgrading / deploy notes
- Run `uv run python scripts/migrate_hash_access_keys.py` (options: `--dry-run`, `--drop-plaintext`) to convert stored access keys to hashes. Access keys now use the `rgd_` format and are shown only once, when created.
- Run `uv run python scripts/rebuild_faiss_index.py --all` once after deploying (or `--agent-id <id>` for one agent). It drops stale vectors from replaced documents and is safe while the backend runs. Older indexes are converted on first update or removal (or by `rebuild_faiss_index.py`).
- Provider API keys are never returned by the API.

## Further reading
- [Deploy on a server](docs/manuals/deploy_on_server.md)
- [Async document upload](docs/manuals/async_document_upload.md)
- [Local translation setup](docs/manuals/local_translation_setup.md)
- [Piper TTS voices](docs/manuals/tts_piper_voice_setup.md)
- [Architecture](docs/architecture/architectural_design.md)

## Contributors

<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/tobiasfremming">
          <img src="https://github.com/tobiasfremming.png?size=100" width="100px;"/><br />
          <sub><b>Tobias Fremming</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/haluboi">
          <img src="https://github.com/haluboi.png?size=100" width="100px;"/><br />
          <sub><b>Halvor Heien Førde</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/peter-olai">
          <img src="https://github.com/peter-olai.png?size=100" width="100px;"/><br />
          <sub><b>Peter Olai Johnsen</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/hallvardvo">
          <img src="https://github.com/hallvardvo.png?size=100" width="100px;"/><br />
          <sub><b>Hallvard Vatnar Olsen</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/erikleblanc">
          <img src="https://github.com/erikleblanc.png?size=100" width="100px;"/><br />
          <sub><b>Erik Le Blanc Pleym</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/selinyo">
          <img src="https://github.com/selinyo.png?size=100" width="100px;"/><br />
          <sub><b>Selin Yuki Øzkan</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/snobohle">
          <img src="https://github.com/snobohle.png?size=100" width="100px;"/><br />
          <sub><b>Erik Olsen Bøhle</b></sub>
      </a>
    </td>
  </tr>
</table>
