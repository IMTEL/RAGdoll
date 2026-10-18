# Review fixes: implementation plan

Spec: `docs/2026-10-18-weakness-review-fix-plan.md` (the verified findings, with evidence). Section numbers like "§0.1" refer to that document. Line numbers there were recorded **before** the ruff formatting commit `8bcf24d`, so treat them as approximate and locate code by symbol.

## Global constraints (every task)

- **Repo and branch:** `C:\Users\tobia\Documents\RagdolProd\RAGdoll`, branch `review-fixes`. Never push, never switch branches, never rebase.
- **Commits:**
  - Use the repo's configured git identity (`tobiasfremming`).
  - **Do not add any `Co-Authored-By` or other Claude/AI attribution lines.**
  - Write a plain imperative subject line, plus a body explaining why.
  - One or more commits per task.
- **Never stage these uncommitted WIP files:** `docs/manuals/user_guide_agents_external_access.md`, `src/access_service/factory.py`, `src/models/chat/message.py`. Always `git add <explicit paths>`, never `git add -A` / `.` / `-u`. If your task truly requires editing one of these files, stop and report BLOCKED.
- **Lint:**
  - The repo's `.ruff.toml` has `fix = true` until Task 1 changes it. Always run `uv run ruff check --no-fix <paths you changed>`, then `uv run ruff format <paths you changed>`.
  - Never run ruff over the whole tree with fixes on.
  - New code must not add new ruff findings in the files you touch.
- **Tests:**
  - Use pytest via `uv run pytest <test paths> -q --no-cov -p no:cacheprovider`, with env `RUNNING_TESTS=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`.
  - After Task 1, tests use the in-memory Mongo fixture and need no MongoDB server.
  - Before Task 1 is complete, do not run tests that need MongoDB.
  - Write the failing test first, see it fail, then implement (TDD).
- **Machine is memory-constrained:**
  - Do NOT use Docker. Do NOT start servers.
  - Do NOT load Whisper, Piper, sentence-transformers, spaCy or real LLM/embedding clients in tests; stub them.
  - Do not run the full suite more than once per task; prefer targeted test files.
  - No real network calls in tests.
- **Code style:** match the surrounding code (FastAPI, pydantic v2, pymongo DAOs, `logging.getLogger(__name__)`). Keep changes scoped to the task. Do not refactor unrelated code.
- **Off limits:**
  - `src/graph/` is unfinished code that imports modules which don't exist. Do not touch it.
  - `tests_deprecated/` is only touched in Task 17.

---

### Task 1: Test infrastructure without MongoDB, and lint safety (§5.1)

**Goal:** tests run locally and in CI without a MongoDB server, and linting never rewrites files by surprise.

1. `.ruff.toml`: change `fix = true` to `fix = false`.
2. Add `mongomock` to the `dev` dependency group (`uv add --dev mongomock`, and commit `uv.lock`).
3. Make tests independent of a real Mongo:
   - In `tests/conftest.py`, patch `pymongo.MongoClient` with `mongomock.MongoClient` before any `src` module is imported. A module-level patch in conftest, or an autouse session fixture that runs before imports, both work. Pick whichever actually works given that `src/globals.py` builds DAOs at import time.
   - If import-time DAO construction makes this impossible, make DAO construction lazy instead (for example `functools.cache` getters in `src/globals.py` and `src/rag_service/dao/factory.py`), keeping the public names other modules import working.
4. `src/config.py`: under `RUNNING_TESTS`, `MONGODB_DATABASE` must come from `MOCK_MONGODB_DATABASE` (default `test_database`), never from the production variable. This prevents tests writing into a developer's real database.
5. `.github/workflows/tests.yml`:
   - Lint uses `ruff check --no-fix`.
   - Add a `uv lock --check` step to the lint job.
6. `.github/workflows/deploy.yml` (if present):
   - Remove `secrets.*` from any step-level `if:`. Move the check into an env var or a job-level condition that GitHub allows.
   - Make deployment depend on lint and tests passing (`needs:`, or a `workflow_run` trigger on "Tests and Lint" with conclusion success). Keep the existing deploy steps unchanged.
7. Run the whole `tests/` suite once with no MongoDB running.
   - Record pass/fail per file in the report.
   - Tests that were already broken for unrelated reasons may be marked `@pytest.mark.skip(reason=...)` only if they fail at the base commit too. List each one in the report.

**Acceptance:**
- `uv run pytest tests -q --no-cov` completes (no hang) with no Mongo server present.
- `ruff check` no longer modifies files.

### Task 2: Fix inverted access-key check and auth wiring (§0.1)

**Goal:** chat endpoints accept only valid access keys or authorized users.

1. `src/utils/path_dependencies.py` `has_agent_call_access`:
   - Raise 403 when `access_service.authenticate(...)` returns **False**, not when it returns True.
   - On the user path, call auth with the correct argument order (`auth_service.auth(authorize, agent_id)`; check the signature), or reuse the existing `require_user_has_agent_access` logic.
2. `get_optional_user`:
   - Only fall back to treating the bearer string as an access key when it is not JWT-shaped (three dot-separated base64url segments).
   - A JWT that fails validation (expired, bad signature, Keycloak unreachable) must produce 401, not silently become an access-key attempt.
3. `src/routes/chat.py` `ask`: the no-`access_key` path currently passes a `Request` to `get_auth`. Extract the bearer credentials from the `Authorization` header correctly; missing header → 401.
4. `src/routes/agents.py`:
   - The access-key route that calls `authenticate(agent, ...)` with an `Agent` object must pass the agent id.
   - Register the static GET routes (`/agent-info-by-accesskey`, `/new-accesskey`, `/chat-accesskey`, and any other literal path under the same prefix) **before** `GET /agents/{agent_id}` so they are reachable.
   - Fix the user-search filter that compares `user.id != user.id`, which is always False. Read the intent from the code: probably "exclude the requesting user".
5. **Tests**, using the real `AccessService` with the existing mock agent DAO (not `MockAccessService`), through FastAPI `TestClient` with the pipeline stubbed:
   - valid key → 200
   - wrong key → 403
   - expired key → 403
   - no credentials → 401 or 403
   - user without access → 403 or 404, whichever the existing auth service raises
   - owner or collaborator user → 200
   - static agent routes are reachable (not swallowed by `/{agent_id}`)

### Task 3: Small chat and LLM correctness bugs (§2.5)

1. `src/routes/chat.py`:
   - Every `HTTPException(detail={...})` that builds a **set** (`{str(e)}`, `{f"..."}`) must pass a string.
   - Error paths must keep their intended status codes (for example LLM errors keep `e.status_code`).
2. `ask_transcribe_with_speech`:
   - Synthesize speech from the agent's answer text (the nested pipeline response's `response` string), not from the wrapper dict.
   - Return a single JSON object (no trailing-comma tuple): the transcribe response plus a `speech` key.
   - Make `/askWithSpeech` consistent if it has the same shape problem.
3. `src/llm.py` `OpenAILLM`:
   - Messages must use the key `content`, not `text`.
   - When the API returns `None` content, raise the module's `LLMGenerationError` (or the equivalent existing error) instead of returning `None`.
4. `src/pipeline.py`: the main answer generation call must pass the agent's `llm_max_tokens` and `llm_temperature`, the same way the retrieval-query rewrite call already does.
5. **Tests** (stub STT, pipeline, TTS and the HTTP clients):
   - an LLM error surfaces with its status code and a string detail
   - `askTranscribeWithSpeech` returns an object with `transcription`, `response` and `speech`
   - `OpenAILLM` builds `content` messages (mock the client)
   - the pipeline passes max_tokens and temperature to `generate`

### Task 4: Stop logging prompts and secrets (§0.2)

1. Remove the `print(...)` calls in `src/pipeline.py` that output the full prompt, the rewritten query and document ids. Also remove the prints in `src/rag_service/dao/context/mongodb_context_dao.py`. Where a trace is useful, replace it with `logger.debug` that logs only counts or lengths (for example "prompt assembled: 3 chunks, 2140 chars").
2. `src/constants.py` `LOGGING_CONFIG`:
   - Root level INFO (configurable via `LOG_LEVEL`, default `INFO`).
   - `pymongo`, `httpx`, `httpcore`, `openai`, `urllib3`, `google`, `faiss` loggers at WARNING.
   - Replace the unbounded `FileHandler("app.log")` with stdout only by default. Write a file only when `LOG_FILE` is set, and then use `RotatingFileHandler` (10 MB × 5).
3. Add `LOG_CONVERSATIONS` (default false) to `src/config.py`.
   - When true, the pipeline may log the user message and the response at INFO through a small redaction helper that masks anything that looks like a key or token (`rgd_…`, `sk-…`, long base64/hex runs, `Bearer …`).
   - When false, no conversation content is logged anywhere.
   - Document both settings in `.env.example`.
4. **Test:**
   - Run `assemble_prompt_with_agent` with stubbed LLM, embedding and DAO, with a sentinel string in the chat log and in a retrieved chunk. With `caplog` at DEBUG plus `capsys`, assert the sentinel appears in neither.
   - Unit-test the redaction helper.
5. Do not delete any `app.log` files. Do not touch `src/graph/`.

### Task 5: Lock down unauthenticated endpoints (§0.3)

1. `src/routes/debug.py`:
   - `GET /api/debug/logs` requires an authenticated user and returns only entries for agents that user owns or collaborates on.
   - The Keycloak bootstrap route stays gated by `AUTH_BOOTSTRAP_ENABLED` and additionally requires an authenticated user.
2. Add a dependency `require_user_or_access_key` in `src/utils/path_dependencies.py`. It accepts either a valid user JWT or a valid access key for the agent named in an `X-Agent-Id` header, and returns 401/403 otherwise. Apply it to:
   - `/api/chat/transcribe`, `/api/chat/tts`, `/api/chat/tts/warmup`, `/api/chat/stt/warmup`
   - every route in `src/routes/translation.py` and `src/routes/transliteration.py`
   - `/upload/status/{task_id}`
   - Keep request and response bodies unchanged.
3. Add a guard test that iterates over `src.main.app.routes` (APIRoute only) and asserts that a request without credentials gets 401/403/422-for-missing-auth on every route, except an explicit allowlist: `/`, `/ping`, `/docs`, `/docs/oauth2-redirect`, `/redoc`, `/openapi.json`, plus any route that already authenticates by its own header (the `/api/progress/*` routes use an `access_key` header; allowlist them with a comment saying why).
4. Do not add rate limiting in code. Instead, add a commented `limit_req` example for the translate, tts and transcribe locations in `nginx.conf`, noting it is untested.

### Task 6: Chunk id uniqueness and the cross-agent lookup (§1.2, quick part)

1. Find where chunk ids are generated (`sha256(f"{file_path}_{i}")` in `src/scraper_service/scraper.py`, consumed in `src/context_upload.py`). New chunks get globally unique ids, for example `f"{document_id}:{chunk_index}:{uuid4().hex[:12]}"`. Make sure `document_id` is available where the id is created, or assign the id in `context_upload.py`.
2. `src/rag_service/dao/context/hybrid_search.py` `vector_search`: the Mongo lookup by `chunk_id` must also filter by `agent_id`.
3. In the same function, remove the fallback that returns chunk text from FAISS metadata when Mongo has no row. A vector with no Mongo row is skipped, because Mongo is the source of truth.
4. **Tests:**
   - Two agents upload documents with the same filename and different text. Searching agent A never returns agent B's text.
   - Re-uploading a document yields new chunk ids, and no Mongo duplicates are left behind.

### Task 7: FAISS cold start must never delete an index (§1.1)

All changes in `src/rag_service/dao/context/faiss_vector_store.py` and `hybrid_search.py`.

1. `FAISSVectorStore` and the manager's `get_store`/`search`:
   - `embedding_dim` becomes optional.
   - When an index file exists, adopt its dimension (`index.d`).
   - On a mismatch between a requested dimension and the file, **never delete**: raise a clear error (custom `IndexDimensionMismatchError`).
2. If loading fails for any other reason (corrupt file), rename the files to `*.corrupt-<timestamp>` and raise. Never silently create an empty index that would later overwrite the file.
3. `get_store`: if a cached store's dimension differs from the requested one, raise instead of returning the mismatched store.
4. `vector_search` passes `embedding_dim=len(query_embedding)`.
5. Save atomically: write to a temp file in the same directory, then `os.replace`. Do this for both the index and the metadata.
6. Add `rebuild_index(agent_id)` on the manager. It rebuilds the agent's FAISS store from the Mongo `context` collection rows, which store `embedding`. Expose it as a CLI script `scripts/rebuild_faiss_index.py` that takes `--agent-id` or `--all`. **Do not run it** against any real database.
7. Make `insert_context` in `mongodb_context_dao.py` stop swallowing FAISS add failures. Log the error and re-raise, so uploads report failure.
8. **Tests** (temp dir for `FAISS_INDEX_DIR`, small dims such as 8 and 16):
   - Add 16-dim vectors, clear the manager cache, search with a 16-dim query → results come back and the file still exists.
   - Search with a different dimension → error, and the file is still intact.
   - A corrupt file is renamed and an error is raised.
   - The rebuild from mock Mongo rows works.

### Task 8: Surface retrieval failures and fix the Gemini SDK usage (§1.4)

1. `src/rag_service/embeddings.py` and `src/llm.py` (Gemini parts):
   - Verify whether `from google import genai` plus `genai.configure` / `genai.embed_content` / `GenerativeModel` exist in the installed `google-genai` package (`uv run python -c ...`, no network).
   - If they don't, port both to the `google-genai` client API: `genai.Client(api_key=...)`, `client.models.embed_content(...)`, `client.models.generate_content(...)`. Preserve the output shapes (a list of floats for embeddings; text for generation) and the existing error mapping to `EmbeddingAPIError` and `LLMAPIError`.
2. `src/pipeline.py` `assemble_prompt_with_agent`:
   - Replace the catch-all around retrieval with handling of the expected error types: embedding errors, DAO errors, and `IndexDimensionMismatchError` from Task 7.
   - Retrieval produces `retrieval_status` in {`ok`, `empty`, `error`} plus `retrieval_error` (a string, no secrets).
3. Add the agent setting `on_retrieval_error: Literal["fail", "answer_without_context"] = "fail"` in `src/models/agent.py`.
   - With `fail`, the request ends with HTTP 503 and detail `"Knowledge retrieval failed"`.
   - With `answer_without_context`, generation continues with an added instruction: the character has no reference material for this answer and must say it isn't sure rather than invent facts.
   - Agents stored without the field default to `fail`.
4. The response includes `retrieval: {"status": ..., "error": ...}`. Keep `num_context_retrieved`.
5. **Tests:**
   - an embedding stub that raises → 503 under `fail`
   - the same stub under `answer_without_context` → 200 with `retrieval.status == "error"` and the uncertainty instruction present in the prompt passed to the LLM stub
   - `status == "empty"` when nothing matches
   - the Gemini embedding and LLM classes can be constructed with a dummy key without AttributeError (mock the client, no network)

### Task 9: Unblock the event loop (§2.1)

1. `src/routes/chat.py`: no blocking work (LLM, embeddings, Mongo, FAISS, Whisper, Piper) may run directly on the event loop.
   - Either convert handlers to plain `def` (FastAPI runs them in its threadpool), or keep `async def` and wrap blocking calls with `await run_in_threadpool(...)`.
   - Where one handler calls another (for example `ask_with_speech` → `ask`), factor out sync helpers so nothing blocking runs inside `async def`.
2. Add timeouts:
   - `requests.post(..., timeout=LLM_TIMEOUT_SECONDS)` in `src/llm.py`.
   - Pass a timeout to the OpenAI and Gemini clients.
   - The embedding HTTP calls get the same timeout.
   - Add `LLM_TIMEOUT_SECONDS` (default 60) to `src/config.py` and `.env.example`.
3. Bound heavy CPU work: wrap Whisper transcription and Piper synthesis in a module-level `threading.BoundedSemaphore`, sized by `STT_MAX_CONCURRENCY` / `TTS_MAX_CONCURRENCY` (default 2 each, in config).
4. **Test:**
   - With the pipeline stubbed to `time.sleep(1.5)`, send `/api/chat/ask` and `/ping` concurrently using `httpx.AsyncClient` with `ASGITransport`. Assert `/ping` completes in under 0.5 s.
   - Assert the LLM `requests.post` gets a timeout (mock).

### Task 10: Upload temp-file collisions (§4.2)

All changes in `src/routes/upload.py` and `src/context_upload.py`.

1. Save each upload to a unique path, `temp_files/<task_id><suffix>` (or `tempfile.mkstemp(dir=temp_files, suffix=...)`). Never use the client's filename on disk.
2. Pass the original filename (`Path(file.filename).name`) explicitly as `document_name` into `process_file_and_store` (new parameter), so stored document names stay the same.
3. Stream the upload to disk with `shutil.copyfileobj` and enforce `MAX_UPLOAD_MB` (default 50, in config) → 413 when exceeded. Clean up the partial file.
4. Serialize processing per `(agent_id, document_name)` with a lock dict guarded by a lock. This is in-process only; note the multi-process limitation in a comment.
5. On startup, delete files in `temp_files` older than 1 hour (FastAPI lifespan or startup event in `src/main.py`).
6. **Tests:**
   - Two uploads named `x.txt` with different content, processed with `process_file_and_store` stubbed to record what it read → each task reads its own content.
   - An oversize upload → 413, and no file is left behind.

### Task 11: FAISS id-mapped inner-product index, document removal, correct scores (§1.2 full, §1.3)

1. In `faiss_vector_store.py`:
   - New indexes use `faiss.IndexIDMap2(faiss.IndexFlatIP(d))` over L2-normalized vectors.
   - Each vector gets a stable int64 id. Store the `int_id → chunk_id` map, and the `document_id` per id, in the metadata file. Stop relying on dict insertion order.
   - Search scores are cosine similarity, taken directly from inner product and clamped to [-1, 1]. Thresholds keep their meaning (cosine).
2. Legacy `IndexFlatL2` files still load:
   - Search on them uses the corrected conversion `score = 1 - d/2` (not `d**2`), or they can be converted by `rebuild_index`.
   - Log a warning recommending a rebuild.
   - Make `rebuild_index` (Task 7) always produce the new format.
3. Add `remove_document(agent_id, document_id)` using `remove_ids`, and expose it on the context DAO (`delete_by_document_id`).
   - Call it from the document replace path in `src/context_upload.py`, which should embed and insert the new chunks first, then remove the old ones.
   - Call it from document deletion in `src/rag_service/dao/document/mongodb_document_dao.py`.
4. **Tests:**
   - cosine 0.8 between two unit vectors → score 0.8 ± 1e-5
   - upload "alpha", re-upload "beta", search → no "alpha" in the results, and the store size equals the Mongo chunk count
   - delete a document → its vectors are gone
   - a legacy L2 index still loads and searches

### Task 12: Hash access keys (§4.3)

1. New key format: `rgd_<keyid8>_<secret>`, with `secret = secrets.token_urlsafe(32)`. Store `key_id`, `key_hash` (SHA-256 hex of the full key, compared with `hmac.compare_digest`) and `prefix` (the first 12 characters, for display). Never store the plaintext.
   - Ruling recorded: SHA-256 instead of argon2. The keys are 256-bit random, so slow hashing adds nothing, and lookup stays fast.
2. `AccessService.generate_accesskey`:
   - Returns the plaintext once, through a dedicated response model `AccessKeyCreated`.
   - `authenticate` parses `key_id`, looks up that key, compares the hashes, checks expiry and updates `last_use`.
   - Legacy keys that are stored in plaintext keep working (compare against `key` if present, using `hmac.compare_digest`) until migrated.
3. Responses:
   - `AccessKey.key` is never serialized in API responses.
   - Add an `AgentPublic` response model, used by every `/agents` endpoint that returns agents, that excludes `access_key` secrets, `llm_api_key` and `embedding_api_key`.
   - `view_once=False` / "show anytime" behaviour is removed. `chat-accesskey` mints a new short-lived key (24 h) instead of returning an existing one.
   - `GET /api-keys/{key_id}` stops returning the decrypted provider key. Return metadata only, with a masked value.
4. Add the migration script `scripts/migrate_hash_access_keys.py`. For every stored key with `key` set and no `key_hash`, set `key_hash` = SHA-256 of the existing key and set `prefix`. Leave `key` in place until run with `--drop-plaintext`. It must be idempotent and support `--dry-run`. **Do not run it.**
5. RAGdollConfig (`C:\Users\tobia\Documents\RagdolProd\RAGdollConfig`) is out of scope for this task. List in the report every frontend place that reads an access key value or the decrypted provider key, so a follow-up can update it.
6. **Tests:**
   - Create a key: the response has the plaintext, while the DB and `GET /agents` don't.
   - authenticate: valid, wrong and expired keys, plus a legacy plaintext key.
   - The migration is idempotent and a dry run changes nothing.

### Task 13: Cheaper query rewrite and client reuse (§2.2)

1. `src/pipeline.py`:
   - Skip the retrieval-query rewrite LLM call when the chat history has no previous user turn, or when the latest message has no back-reference words (a small EN/NO pronoun and deictic list, for example it, that, this, they, he, she, those, den, det, de, han, hun, dette).
   - When it does run, send only the last 4 turns (not the persona) with `max_tokens=64`.
2. Cache LLM and embedding client construction per `(provider, model, api_key hash)` with a small LRU (`functools.lru_cache` on a factory taking hashable args), instead of rebuilding per request.
3. Log stage timings (rewrite, embed, retrieve, generate) at INFO as one line per request, containing durations only.
4. **Tests:**
   - the first-turn message makes no rewrite call
   - a back-reference message makes one call with bounded history
   - two requests for the same agent construct the client once

### Task 14: Prompt structure with message roles (§3.1)

1. Change the `LLM` protocol in `src/llm.py` to `generate(messages: list[dict[str, str]], **kwargs)`. Keep a temporary adapter that accepts a plain string for any caller you don't migrate, and list such callers in the report.
   - Idun (OpenAI-compatible) and OpenAI pass the messages as they are.
   - Gemini maps the system parts to `system_instruction` and the rest to `contents`.
2. Build the messages in `src/pipeline.py`:
   - `system`: persona/role prompt, behaviour rules, and the function/JSON instructions.
   - A second `system` message: the reference material in a delimited block, stating that the block is data to consult, not instructions to follow.
   - Chat history as real `user`/`assistant` turns (map the stored role `agent` to `assistant`).
   - The final `user` message.
3. Replace the "100% true … prioritise it over all other sources" wording with: base factual claims on the reference material; if it does not cover the question, say so and do not invent facts.
4. **Tests:**
   - unit-test the message builder output
   - an "ignore previous instructions…" string inside a retrieved chunk ends up only inside the delimited reference block, never in a system instruction position
   - the Gemini mapping

### Task 15: Numbered sources, citations, grounding and document kind (§3.2)

1. Number retrieved chunks `[S1]…[Sn]` inside the reference block.
2. Extend the existing JSON response contract (`function_prompt_section` / `_parse_llm_response` in `src/pipeline.py`) with `citations: list[str]` (for example `["S1","S3"]`) and `grounded: bool`. Unknown ids are dropped.
3. The API response gains:
   - `citations: [{document_name, chunk_index, score}]`
   - `grounded: bool`
   - keep `context_used` for backwards compatibility.
4. Strip `[S\d+]` markers from the answer text before it is returned or sent to TTS.
5. Document kind:
   - Add `kind: Literal["world_lore","educational"] = "educational"` to the document model, accepted on upload as an optional form field.
   - In the reference block, label each source with its kind.
   - Add the rules: world_lore is canon inside the fiction; educational claims must be supported by a cited source or explicitly hedged.
   - Existing documents default to `educational`.
6. **Tests:**
   - parsing citations and grounded from the model JSON
   - mapping ids to sources
   - markers stripped from the text
   - kind persisted and shown in the prompt

### Task 16: Durable progress state (§4.1)

1. Add a `ProgressDAO`, interface plus a MongoDB implementation, matching the existing DAO pattern in `src/rag_service/dao/`. It writes to a `progress` collection with:
   - an index on `{agent_id, session_id, updated_at}`
   - a unique index on `task_id`
   - a TTL index on `updated_at`, `PROGRESS_TTL_HOURS` (default 24)
   - plus a separate `upload_tasks` collection for upload status.
2. Replace `src/utils/global_logs.progress_log` usage in `src/routes/progress.py`, `src/routes/upload.py`, `src/routes/chat.py` (`_attach_recent_progress` / `get_recent_progress_for_session`) and `src/routes/debug.py` with the DAO. Keep the endpoints' request and response shapes.
3. Fix the `completet_at` typo, keeping backwards-compatible reading of old entries if any are persisted. Upload entries no longer get pruned instantly, and a session id of `"default"` no longer crashes chat.
4. Add a mock implementation for tests, following the existing mock DAO pattern.
5. **Tests:**
   - progress survives "restart" (new DAO instance, same mongomock client)
   - upload status stays available after a progress GET
   - a `"default"` session doesn't crash chat

### Task 17: Remove dead streaming code and deprecated tests; fix README (§2.4, §5.2)

1. Delete `src/streaming_ws.py`, `src/web_socket_sandbox.py` and `tests_deprecated/` (the whole folder). First grep for imports or references to them (including Dockerfile, compose and docs) and remove those.
2. README.md:
   - Clone URL `https://github.com/IMTEL/RAGdoll.git`.
   - Replace the pip/requirements steps with `uv sync --dev` and `uv run ...`, and the docker compose instructions (`docker compose -f docker-compose.local.yml up --build`).
   - Document `RUNNING_TESTS` and how to run tests (no Mongo needed after Task 1).
   - Document Keycloak, Mailpit and `SMTP_*`, `AUTH_BOOTSTRAP_ENABLED`, `LOG_CONVERSATIONS`, `LOG_LEVEL` / `LOG_FILE`.
   - Remove links to files that don't exist.

### Task 18: Faster spoken replies: parallel sentence TTS (§2.3, short-term part)

1. In `src/routes/chat.py` / `src/tts.py`, for the speech endpoints:
   - Split the answer into sentences (a simple regex on `.!?` followed by whitespace, keeping abbreviations intact is best effort).
   - Synthesize the sentences in parallel within the Task 9 TTS semaphore bound.
   - Return `speech_segments: [{text, audio_base64, sample_rate}]` in order, alongside the existing single `speech` field, which is concatenated for backwards compatibility.
2. **Tests:** with a stubbed synthesizer, the order is preserved, segments match the sentences, and the concatenated `speech` matches the previous behaviour for a one-sentence answer.
3. Ruling recorded: the full token-streaming WebSocket path with barge-in (§2.3 proper) is out of scope for this plan. It needs a client protocol decision with the XR/Unity side.
