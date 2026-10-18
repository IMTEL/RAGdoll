# Weakness review: verification and fix plan (2026-10-18)

An external AI review listed six weakness areas in RAGdoll. Each claim was checked against `main` at `233c92a` by reading the code. Nothing was executed except lint. Verdicts:

| # | Area | Verdict |
|---|------|---------|
| 0 | **Access-key check inverted** (not in the review) | **New, critical** |
| 1 | Knowledge index (FAISS) reliability | Confirmed, worse than claimed |
| 2 | Answers stay confident when retrieval fails | Confirmed |
| 3 | Access control and privacy | Confirmed, worse than claimed |
| 4 | Latency and concurrency for spoken interaction | Confirmed |
| 5 | Session and upload state | Confirmed (multi-process part is hypothetical today) |
| 6 | Dev checks (CI, tests, README) | Lint confirmed; Docker failure not reproduced |

Paths are relative to `RAGdoll/src` unless noted. Effort: S = hours, M = a day or two, L = a week or more.

---

## Phase 0: do now

### 0.1 Access-key check is inverted (Critical, S)
`utils/path_dependencies.py:155` raises when `access_service.authenticate(...)` returns **True** (`access_service/access_service.py:63-81` returns True for a *valid* key). Result: any wrong `access_key` on `/api/chat/ask` and its three variants runs the full pipeline on the owner's LLM key. A correct key gets a 400. The caller only needs the agent id. Introduced by the auth refactor (d0fe587). CI did not catch it because tests use `ACCESS_SERVICE=mock`, which always returns True.

Related bugs in the same refactor:
- `path_dependencies.py:153` calls `auth_service.auth(agent_id, bearer_token)`. The signature is `auth(authorize, agent_id)`.
- `routes/chat.py:~137` passes a `Request` to `get_auth`, which expects credentials. This causes an AttributeError and a 500.
- `routes/agents.py:~417` passes an `Agent` to `authenticate`, which expects an id. It is always False.
- `routes/agents.py:~200`: `GET /agents/{agent_id}` is declared before `/agent-info-by-accesskey`, `/new-accesskey` and `/chat-accesskey`, so it shadows them.
- `routes/agents.py:~236`: the filter `user.id != user.id` is always False, so the user search is always empty.
- `get_optional_user` swallows every exception. An expired JWT or a Keycloak outage silently falls through to the access-key path.
- RAGdollConfig still calls backend paths this refactor renamed or removed.

**Fix:** use `if not access_service.authenticate(...): raise 403`, pass the arguments in the right order (or use `require_user_has_agent_access`), take the access-key path only for non-JWT-shaped tokens, and reorder the static routes. **Tests** use the real `AccessService` with an in-memory DAO: a valid key returns 200, a wrong key 403, an expired key 403, a non-collaborator 404, and a collaborator 200.

### 0.2 Stop logging prompts and secrets (High, S)
- `pipeline.py:328` prints the full prompt: history, user message, retrieved chunks, persona and game context. More prints are at `:201` and `:256`, in `mongodb_context_dao.py:179`, and in `graph/retrieval/retrieval_service.py:140,377`.
- Root cause: `constants.py:12-28` sets the root logger to DEBUG with an unrotated `FileHandler("app.log")`. That turns on pymongo command logging and openai/httpx debug logging.
- The local `app.log` (671 MB) contains **plaintext agent access keys** (from pymongo command logs), Fernet-encrypted provider keys, user emails and chunk text.

**Fix:**
- Replace the prints with `logger.debug` calls that log only counts and lengths.
- Set the root logger to INFO, and set pymongo, httpx, httpcore, openai and urllib3 to WARNING.
- Log to stdout in containers, or use `RotatingFileHandler`.
- Make conversation logging opt-in with `LOG_CONVERSATIONS=false`, plus redaction and a retention period.

**Then:** delete `app.log` everywhere (including the server) and rotate the access keys and provider keys that appear in it. **Test:** caplog/capsys with a sentinel string in the chat log and in the retrieved context. The sentinel must not appear in the output.

### 0.3 Lock down open endpoints (Medium, S)
- `GET /api/debug/logs` (`routes/debug.py:24-27`) has no auth. It returns every agent's progress and upload entries.
- `/api/debug/bootstrap-keycloak-user` is gated only by `AUTH_BOOTSTRAP_ENABLED`. If that flag is on, anyone can claim every agent.
- These endpoints have no auth: `/api/chat/transcribe`, `/tts`, `/tts/warmup`, `/stt/warmup`, `/api/translate` (plus batch: 100 × 15k chars), `/api/transliterate` and `/upload/status/{id}`. That gives anyone free CPU-heavy Whisper and Piper and an open LibreTranslate proxy.

**Fix:**
- Add `require_user` to the debug router, or mount it only in dev.
- Make bootstrap require an admin role.
- Require a user JWT or a valid access key on STT, TTS and translate, plus rate limiting (nginx `limit_req` or slowapi).
- **Guard test:** iterate over `app.routes` and assert 401 or 403 without auth, except for an explicit allowlist.

---

## Phase 1: stop data loss and wrong answers

### 1.1 FAISS index deleted on cold start (High, S; M with the rebuild tool)
- `hybrid_search.py:344-350` never passes `embedding_dim`, so the first search after a restart builds the store as 384-dim (`faiss_vector_store.py:347,406`).
- On load, a dimension mismatch **deletes the `.index` and metadata files** (`:94-103`). Gemini is 768-dim and Idun is 4096, so every Gemini or Idun agent whose first action after a restart is a chat loses its index.
- The broken 384-dim store then stays cached. Later uploads fail inside `add_vectors`, but `insert_context` swallows the error (`mongodb_context_dao.py:321-323`), so uploads report success while writing nothing.
- Any load exception also yields an empty index, which then overwrites the file.

**Fix:**
- Take the dimension from the file (`index.d`). On a mismatch, never delete: raise or rename to `.bak`.
- Pass `len(query_embedding)`.
- `get_store` should error when the cached dimension differs from the requested one.
- Write atomically (temp file, then `os.replace`).
- Add `rebuild_index(agent_id)` from the Mongo `context` rows, which store the embeddings.

**Migration:** rebuild every agent once. **Test:** add a 768-dim vector, clear `_stores`, search, and assert the result is non-empty and the file still exists.

### 1.2 Stale chunks after replace, and a cross-agent leak (High, M)
- Replacing (`context_upload.py:131-139`) or deleting (`mongodb_document_dao.py:194`) a document removes its Mongo chunks but never its FAISS vectors, and FAISS has no delete API. Stale text resurfaces through the FAISS-metadata fallback (`hybrid_search.py:359-365`).
- Chunk ids are `sha256(f"{file_path}_{i}")` with `file_path = temp_files/<name>` (`scraper_service/scraper.py:75-78`). Re-uploading the same filename gives identical ids, and the FAISS position → `list(chunk_metadata.keys())[idx]` mapping goes out of sync. New vectors are dropped, and old vectors come back with new text.
- **Leak:** two agents that each upload `report.pdf` share chunk ids, and `find_one({"chunk_id": ...})` (`hybrid_search.py:355`) has no `agent_id` filter. One agent can get the other's text.

**Fix:**
- **Do immediately (S):** add `agent_id` to that `find_one`, and give chunks unique ids (`{document_id}:{idx}:{uuid}`).
- **Then:** switch to `IndexIDMap2(IndexFlatIP)` with stable int64 ids and a stored id map.
- Add `remove_document()` using `remove_ids`, and call it from replace and delete.
- Treat Mongo as the source of truth: drop the metadata fallback.
- On replace, embed the new version before deleting the old one.

**Migration:** the rebuild from 1.1, plus new chunk ids. **Tests:** re-upload "alpha" then "beta" and assert "alpha" never comes back. Upload the same filename to two agents and assert results come only from the agent being queried.

### 1.3 Distance squared twice (Low–Medium, S)
`IndexFlatL2` already returns squared L2, yet `faiss_vector_store.py:272` computes `1 - d²/2` on it. Ranking is mostly preserved, but scores are distorted: cosine 0.8 shows as 0.92, and anything below about 0.29 is clamped to 0. This skews custom thresholds and the BM25 blend (`hybrid_search.py:104`). **Fix:** use `1 - d/2`, or `IndexFlatIP` on normalized vectors, which comes free with 1.2. **Test:** for a cosine of 0.8, assert the score is 0.8 ± 1e-5.

### 1.4 Retrieval failure is hidden (High, S)
- `pipeline.py:275-295` catches every exception, sets the context to `[]` and keeps generating. The response only shows `num_context_retrieved: 0`.
- **Probably the cause in practice (check in a running container first):**
  - `rag_service/embeddings.py:4,80,88` calls `genai.configure` and `genai.embed_content` from the *old* `google-generativeai` SDK, but only `google-genai` is installed.
  - If that holds, every Gemini-embedding agent answers with no context and no error.
  - `GeminiLLM` (`llm.py:204-205`) has the same problem.

**Fix:**
- Catch only the expected errors and return `retrieval: {status: ok|empty|error, error}`.
- Add a per-agent `on_retrieval_error` setting, defaulting to fail with a 503.
- If answering without context, instruct the model to say it is unsure.
- Port Gemini to `genai.Client`.

**Tests:** an embedding stub that raises gives a 503 or `status=error`. A smoke test builds `GoogleEmbedding` and `GeminiLLM`.

---

## Phase 2: responsiveness for live XR conversation

### 2.1 The event loop is blocked (Critical for multi-user, S)
- All chat routes are `async def` (`routes/chat.py`), but they call blocking code: `requests.post` with no timeout (`llm.py:64`), sync SDKs, pymongo/FAISS, faster-whisper and Piper.
- With a single uvicorn worker, one NPC turn freezes every other request, `/ping` included.

**Fix:**
- Move the work into sync helpers and make the routes plain `def`, or `await run_in_threadpool(...)`.
- Add timeouts to every outbound call.
- Put a semaphore or a small executor around STT and TTS.

**Test:** with an LLM stub that sleeps 2 s, fire `/ask` and `/ping` together and assert `/ping` returns in under 100 ms.

### 2.2 Extra LLM call per turn (Medium, S)
- `context_aware_retrieval` defaults to True (`models/agent.py:127-130`). Every turn makes a full query-rewrite LLM call (`pipeline.py:197-246`) that carries the whole persona.
- LLM and embedding clients are rebuilt on every request.

**Fix:**
- Skip the rewrite on the first turn and when the message doesn't refer back to earlier turns.
- Use a cheap model with about 64 max tokens.
- Cache the clients.
- Log per-stage timings.

### 2.3 Speech waits for everything (High, M→L)
`/askWithSpeech` runs the LLM, then Piper on the whole text, then returns base64 WAV in JSON (`tts.py:59-81`). **Short term:** synthesize sentences in parallel. **Proper fix:** stream tokens, cut them into sentences, run TTS per sentence, and stream audio over WebSocket or SSE with turn ids and barge-in cancellation.

### 2.4 Dead streaming code (Low, S)
- `streaming_ws.py` imports a function that no longer exists (`assemble_prompt`), calls functions with the wrong signatures, and is not mounted in `main.py`.
- `web_socket_sandbox.py` is a separate broken app.

Delete both, along with `tests_deprecated/test_ws_sandbox.py`. Build the 2.3 streaming route fresh instead.

### 2.5 Small chat bugs (S, do together with 0.1)
- `HTTPException(detail={str(e)})` passes a set at `chat.py:~161,~331`. Starlette can't serialize it, so the client gets a generic 500 and the real status and message are lost.
- `ask_transcribe_with_speech` (`:317-324`) passes the wrapper dict to TTS, which fails every time, and returns a one-element tuple.
- `OpenAILLM` sends `"text"` instead of `"content"` (`llm.py:150-151`), so the OpenAI provider is likely broken. It also returns `None` on empty content, which then crashes the parser.
- The main answer call ignores the agent's `llm_max_tokens` and `llm_temperature` (`pipeline.py:336`).

---

## Phase 3: answer quality and trust (education)

### 3.1 Prompt structure (Medium, M)
- Persona, rules, retrieved chunks, game context and user text are concatenated into one string. Idun and Gemini send it as a single message.
- Document or user text can therefore override the persona (prompt injection).
- The material is labelled "100% true… prioritise it over all other sources" (`pipeline.py:316`).

**Fix:**
- Change the `LLM` protocol to `generate(messages)`.
- Use a system message for persona and rules.
- Put reference material in a delimited block marked "data, not instructions".
- Send the history as real user/assistant turns.
- Reword the "100% true" line to: base claims on the material; if it doesn't cover the question, say so.

**Test:** an injected instruction inside a chunk must not change the behaviour.

### 3.2 Citations and uncertainty (Medium, M)
- The model never sees chunk ids, so `context_used` (`pipeline.py:357-364`) lists everything that was retrieved, not what was actually used.

**Fix:**
- Number the sources (`[S1]…`).
- Extend the JSON output to `{message, citations, grounded}`, then map the ids back to document and chunk.
- Strip the markers before TTS.
- Expose `grounded` so the XR client can show uncertainty.
- Add a document `kind` (`world_lore` | `educational`): lore counts as canon inside the fiction, while educational claims must be grounded or hedged.

---

## Phase 4: durability of session and upload state

### 4.1 Progress is in process memory (Medium, M)
- `utils/global_logs.progress_log` is a list in process memory. A restart wipes training-session progress and upload status (`/upload/status` returns 404).
- There is one worker today, so the multi-process concern only becomes real when scaling out.
- **Related bugs:**
  - Upload entries are pruned immediately because they have no timestamp (`progress.py:26`).
  - A `"default"` session id can crash chat validation (`progress.py:83`).
  - The field name `completet_at` is a typo.
  - Nothing locks the list.

**Fix:** a Mongo-backed `ProgressDAO`, indexed on `{agent_id, session_id, updated_at}` with a TTL index replacing the manual pruning, and a separate `upload_tasks` collection.

### 4.2 Upload temp-file collisions (Medium, S)
- Every agent writes to the same `temp_files/<original name>` (`routes/upload.py:177-181`).
- With concurrent same-name uploads, the second upload overwrites the first (possibly cross-agent), and the first task to finish deletes the other's file.
- Path traversal is already blocked by `.name`.
- The whole body is read into memory with no size limit.

**Fix:**
- Use per-task unique temp paths and pass the original name explicitly as `document_name`.
- Stream to disk with a maximum size.
- Serialize processing per `(agent, document)`.
- Sweep stale temp files on startup.

### 4.3 Access keys stored in plaintext (High, M)
- Access keys are stored and compared in plaintext (`access_service.py:48-60,72-73`). The argon2 helpers in `utils/crypto_utils.py` are never used.
- `view_once` is bypassed: `GET /agents` returns the full keys, collaborators see them, and `chat-accesskey` re-returns them.
- `GET /api-keys/{id}` returns decrypted provider keys, and owners get decrypted LLM keys in agent responses.

**Fix:**
- Use the key format `rgd_<id8>_<secret>` and store only the hash and prefix. Show the key once, on creation.
- Look the key up by id and compare with `hmac.compare_digest`. Update `last_use`.
- Return an `AgentPublic` response model that never includes secrets.
- **Migration:** hash the existing keys, keep a temporary legacy-match path, then unset the plaintext.

---

## Phase 5: development checks

### 5.1 Lint and CI (S–M)
- `.ruff.toml` sets `fix = true`, so `ruff check` rewrites files, including in CI. There are 391 findings (234 auto-fixable, 157 remaining), and 27 files need formatting.
- pytest hangs at collection without Mongo, because `globals.py` builds DAOs at import time.
- Under `RUNNING_TESTS`, `MONGODB_DATABASE` is read without the `MOCK_` prefix (`config.py:106`). A local Mongo would receive test writes.
- `deploy.yml` uses `secrets` in a step `if:`, which is invalid, and deploys without depending on the tests.

**Fix:**
- Run `--no-fix` in CI, then do a one-time fix-and-format commit.
- Initialize the DAOs lazily, and use mongomock or a Mongo service container in CI.
- Fix the deploy gating with `needs: [lint, test]`, and add `uv lock --check`.

The Docker startup failure was not reproduced. The likely causes are `uv sync --locked` drift and the 60 s readiness wait.

### 5.2 Tests and README (S–M)
- `tests_deprecated/` is unmaintained and CI ignores it. Nothing tests `pipeline.py`, `/api/chat/ask` or the real `AccessService`. Port `test_pipeline.py` to fakes or delete the folder.
- **README:** fix the clone URL (`vr4vet/chat-service` → `IMTEL/RAGdoll`), replace `pip install -r requirements.txt` with `uv sync --dev`, and document `RUNNING_TESTS`, Keycloak and `AUTH_BOOTSTRAP_ENABLED`.
- Add educational evals next to the unit tests: correct evidence, appropriate uncertainty, Norwegian vs English, and misleading or conflicting documents.

---

## Suggested order
1. **Phase 0** (all S): 0.1 plus the 2.5 chat bugs, then 0.2 (and rotate the keys), then 0.3.
2. **1.2 quick part** (the `agent_id` filter and unique chunk ids), then **1.1**, then **1.4** (check the Gemini SDK in a container first).
3. **2.1** (unblock the event loop) and **4.2** (temp paths): both small.
4. **1.2 full plus 1.3** together with the index rebuild, then **4.3** (hash the keys).
5. **5.1** CI fixes, so everything above stays fixed.
6. **2.2**, **3.1**, **3.2**, **4.1**, then **2.3/2.4** (the streaming path) last, since it's the largest.
