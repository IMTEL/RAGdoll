# Review fixes: release notes and follow-ups (2026-10-18)

Branch `review-fixes`. It implements `docs/superpowers/plans/2026-10-18-review-fixes.md`, which addresses the findings in `docs/2026-10-18-weakness-review-fix-plan.md`.

## Deploy steps
0. **Back up MongoDB and Keycloak first** — see "Back up before every deploy" in `docs/manuals/deploy_on_server.md`. The deploy keeps the data volumes, but the access-key migration below rewrites stored keys.
0b. **Apply the realm fixes by hand** (an existing realm is never re-imported). In the Keycloak admin console, realm `ragdoll` → Clients → `ragdoll-config`:
   - Client scopes → add `basic` as a Default scope (without it tokens have no `sub` and every backend call is 401).
   - Client scopes → `ragdoll-config-dedicated` → Add mapper → By configuration → Audience: included client audience `ragdoll-config`, add to access token ON (without it the backend rejects tokens with "Audience doesn't match").
   - Realm settings → Email: configure SMTP so "Forgot password?" can send mail.
1. Set the new env vars (see `.env.example`). Real email for password reset needs `SMTP_*`; without it, reset emails go to Mailpit.
2. Deploy the backend.
3. Run `uv run python scripts/migrate_hash_access_keys.py --dry-run`, then run it without `--dry-run`. Legacy access keys keep working. Run with `--drop-plaintext` only once you won't roll back.
4. Run `uv run python scripts/rebuild_faiss_index.py --all`. It is safe while the backend is running, and it drops stale vectors left by replaced documents. Retrieval scores are now true cosine values, so a tuned `similarity_threshold` may need lowering.
5. Deploy RAGdollConfig only together with follow-up F1 below.

## Client-facing breaking changes
1. **Speech and translation routes need credentials.** This covers `/api/chat/transcribe`, `/tts`, `/tts/warmup`, `/stt/warmup`, `/api/translate*` and `/api/transliterate*`. Send `Authorization: Bearer <user JWT>`, or an access key plus an `X-Agent-Id` header. A temporary escape hatch is `PUBLIC_SPEECH_ENDPOINTS=true`.
2. **Other routes now need credentials.**
   - `/upload/status/{task_id}` needs credentials, and returns 404 for agents the caller can't access.
   - `/api/debug/logs` needs a logged-in user and returns only that user's agents.
3. **The `ask*` routes authenticate before doing any work.** `askTranscribe` checks credentials before running Whisper.
4. **Access keys have a new format.**
   - Keys look like `rgd_<id>_<secret>`. The plaintext is shown only once, in the creation response. Listings show `prefix` and never `key`.
   - `/chat-accesskey` now issues a new 24 h key on every call.
5. **Provider API keys are never returned.**
   - Agent responses use `AgentPublic`, with `has_*_api_key`, `*_api_key_id` and `*_api_key_invalid`.
   - `GET /api-keys/{id}` returns masked metadata only.
   - The new `llm_api_key_id` / `embedding_api_key_id` fields reference a stored key. Only the owner can set them.
6. **New `/ask` response fields:** `citations[]`, `grounded`, and `retrieval{status, error}`. `[S#]` markers are stripped from the answer text.
7. **Speech responses add `speech_segments[]`,** one per sentence. `speech` is now a single WAV built from the per-sentence audio.
8. **Route reorder.** `GET /agents/agent-info-by-accesskey`, `/new-accesskey` and `/chat-accesskey` were previously shadowed by `/agents/{agent_id}` and are now reachable.
9. **New status codes:**
   - 413: upload larger than `MAX_UPLOAD_MB`.
   - 422: `ask*` request without a body.
   - 503: "Knowledge retrieval failed" under `on_retrieval_error="fail"`, the default.
   - 401/402/429: provider key or quota problems.
   - 401: invalid or expired JWT.
   - 403: wrong access key.
10. **Uploads take an optional `kind`** (`world_lore` or `educational`).

## Follow-ups (not done)
- **F1, RAGdollConfig.** Switch stored-provider-key selection from `raw_key` to `llm_api_key_id` / `embedding_api_key_id`. Update access-key display: no `view_once`, show plaintext only on creation, use `prefix`. Call sites are listed in the Task 12 report.
- **Admin feature.** Designed but not started:
  - Admins are assigned through the `ragdoll-admin` Keycloak realm role.
  - Admins can view and act on all agents and users, and see document metadata only.
  - Admins never see document contents, prompts, conversations, access keys or API keys.
  - Open question: whether to track usage counts.
- **Lint backlog.** 124 legacy ruff findings remain (47 are BLE001). Lint runs but doesn't gate deploy yet. Make it gate deploy once the backlog is clear.
- **Real-provider checks** (not done, because nothing in this work used the network): Gemini `text-embedding-004` availability, Gemini's handling of the new message structure, and how small models follow the JSON citation format.
- **Not in this plan:** a streaming WebSocket path with barge-in, and educational evaluations (Norwegian/English, conflicting documents).
- **Smaller items, deferred:**
  - No cross-process FAISS file lock for multiple workers.
  - `rebuild_index` can race with a live upload.
  - Embedding key errors degrade silently under `answer_without_context`.
  - `/agent-info-by-accesskey` loads every agent (rate-limit it).
  - Chinese sentences aren't split for parallel TTS.
  - The comment in `deploy.yml` still says "Tests and Lint".
