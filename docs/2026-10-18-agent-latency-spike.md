# Agent latency spike: how to make talking to an agent faster (2026-10-18)

A time-boxed investigation of where the time goes in a RAGdoll conversation turn, and which changes would speed it up most. Nothing was changed in the code. The full findings follow below.

## At a glance

- **Today:** a spoken turn takes about **5–12 s (typically ~7 s)** before the user hears anything, because every stage waits for the previous one. A text turn on `/ask` takes about **2.5–8 s**.
- **With streaming** (speak each sentence as soon as the LLM has written it): time to first audio is about **2–3.5 s**.
- **Measured problems:**
  - keyword search reads the agent's entire corpus, embeddings included, on every turn;
  - the query rewrite fires on about 30% of ordinary questions;
  - speech replies carry their audio twice.
- **Not worth changing yet:** the FAISS index type (searches take under 2 ms).

## Recommended order

1. **Quick wins (config or a few lines):**
   - shorter spoken answers: a "1–3 sentences" rule and `llm_max_tokens` ~250;
   - load Whisper and Piper at startup;
   - cap chat history at 10–20 turns;
   - reuse the HTTP connection to Idun;
   - leave the embedding out of the keyword-search query.
2. **Measure first:** replay ~20 real turns with API keys on staging. Extend the `Stage timings` log line with STT, TTS, serialization and total time, so the estimates below become real p50/p95 numbers.
3. **Medium:** fix the query-rewrite trigger (§4.2), send speech audio only once, keep a BM25 index per agent in memory (§4.3).
4. **Large:** an end-to-end streaming endpoint (§4.1). It needs backend and Unity work, and gives the largest gain.

> Numbers marked *(measured)* come from offline benchmarks with stubbed models (mongomock, FAISS, prompt assembly, payload sizes). Whisper, Piper and LLM/embedding timings are typical figures for these models, not measurements of this deployment. There were no API keys and too little free memory to load the speech models. The benchmark scripts were throwaway and are not kept in the repo.

---


## 1. Current latency budget per stage

The estimates come from the code and config. Rows marked **(measured)** were measured offline (§2).
The other rows are typical figures for the given model, size and provider, not measurements of this deployment.

| # | Stage (spoken turn = `/askTranscribeWithSpeech`) | Code | Estimate | Notes |
|---|---|---|---|---|
| 0 | Upload audio (multipart) | `routes/chat.py:363-372` | 50–300 ms | A 5 s, 44.1 kHz, 16-bit WAV is ~440 kB. 16 kHz mono is 160 kB. Opus is ~15 kB. |
| 1 | Parse command, auth | `chat.py:337-340`, `utils/path_dependencies.py:73-201` | 3–15 ms | JWT path: `PyJWKClient` decode, then `get_user_by_provider` and **`set_user` (a Mongo write on every request)** (`auth/auth_provider/keycloak_auth_provider.py:51-69`). Access-key path: `try_get_agent`, then `agent_with_id` (the agent is read twice). |
| 2 | Whisper STT (faster-whisper `base`, int8, CPU) | `transcribe.py:78-96`, `whisper_model.py:32-51` | 0.4–1.5 s | Decoding is fixed at 30 s windows. With `language=None`, Whisper first runs a language-detection pass (`transcribe.py:82`, `_normalize_whisper_language` returns None for auto). `beam_size=1` is already set. VAD is off by default (`docker-compose.yml:152`). `cpu_threads=0` uses the CT2 default (4). `STT_MAX_CONCURRENCY=2` (`config.py:78`). Audio that soundfile cannot decode (webm/ogg) is written to a temp file and decoded with PyAV (`transcribe.py:104-106`). |
| 3 | Second command parse, progress lookup | `chat.py:351`, `chat.py:80-108` | 1–5 ms | One Mongo query. |
| 4 | Provider-key resolution | `pipeline.py:486`, `utils/provider_keys.py:28-72` | 1–5 ms | `get_users_with_agent` plus a Fernet decrypt. |
| 5 | Query rewrite (extra LLM call, conditional) | `pipeline.py:408-467` | 0 or **0.5–1.5 s** | This is a full round trip to the agent's *main* model, with the agent's temperature and `max_tokens=64`. The trigger heuristic fires on **6 of 20** ordinary standalone questions (§2.4): `it`, `this`, `det`, `den` and Norwegian `her` (= "here"). It only fires when there are earlier user turns, which in XR is almost always. |
| 6 | Query embedding | `pipeline.py:687-706`, `rag_service/embeddings.py` | 100–400 ms | One HTTPS round trip. Clients are cached and use pooled connections (openai/genai SDKs). There is no cache of query vectors. |
| 7 | Hybrid retrieval | `hybrid_search.py:64-143` | **50 ms – 1 s+, grows linearly with corpus** | **The keyword half dominates.** `context_collection.find(query)` at `hybrid_search.py:179` has **no projection**: it pulls *every* chunk the role can access, *including the embedding*, on every turn. BM25 then re-tokenizes every chunk twice in pure Python (`:277`, `:285`) while holding the GIL. There is a linear `next(...)` scan per hit (`:208`). The docstring at `:157` says "Results are cached per query", but there is no cache. FAISS itself is under 1 ms. The vector half does N `find_one` calls (`:376`) on a field with no index (`mongodb_context_dao.py:76-88` indexes `agent_id` and `document_id` only). |
| 8 | Prompt assembly | `pipeline.py:779-847` | <1 ms **(measured)** | Prompt *size* is what costs (row 9). Chunks are 4000 chars (`context_upload.py:24`). With top_k=5 (`models/agent.py:81`) the reference block alone is ~5.2k tokens. The history cap of 100 turns (`pipeline.py:110`, `:834`) can add ~5k more. |
| 9 | LLM generation (non-streaming) | `pipeline.py:582`, `llm.py:107-185` (Idun), `:204-223` (OpenAI), `:271-338` (Gemini) | **2–6 s** | Prefill of a 6–11k-token prompt is 0.3–2 s, depending on provider and queue. The full reply is then generated before anything returns: 100–250 output tokens at 40–100 tok/s is 1.5–5 s. `llm_max_tokens` defaults to 1000 (`models/agent.py:73`). The JSON citation scaffolding adds ~20 output tokens (§2.3). **Idun uses a bare `requests.post` (`llm.py:125`)**, so every call opens a new TCP and TLS connection (+1–3 RTT). |
| 10 | Parse reply | `pipeline.py:276-348` | <0.05 ms **(measured)** | |
| 11 | Piper TTS, all sentences | `tts.py:66-89`, `:275-314` | 1–3 s wall | Piper medium on CPU runs at ~0.05–0.15× real time. A 5-sentence, ~30 s answer is 1.5–4.5 s of CPU, spread over 2 workers (`TTS_MAX_CONCURRENCY=2`, `config.py:79`). **Every sentence must finish** before the response is built. Each segment's WAV is decoded again to read its sample rate (`:292`, `:305`) and decoded again to concatenate (`:310`). That costs ~19 ms **(measured)**. |
| 12 | Serialize and download the response | `chat.py:376-379` | 0.3–1.5 s | **The audio is sent twice**: the concatenated `speech` plus every `speech_segments[i]`, both as base64 WAV. A 30 s answer gives a **3.6 MB JSON body** **(measured)**: ~0.6 s at 50 Mbit/s, ~1.4 s at 20 Mbit/s. The client must then parse the JSON and base64-decode it. The web chat only uses `speech` (`RAGdollChat/src/app/external/page.tsx:308-310,725,822`). |
| 13 | First-request penalties | `main.py:47-50` | +2–6 s on the first turn | There is no startup warmup. Whisper and Piper load lazily unless the client calls `/stt/warmup` and `/tts/warmup`. The first LLM and embedding call per key also pays client construction and TLS. |

### Critical path

* **Spoken turn, time to first audio (today).** All stages run in series and the client plays audio only
  after the whole response arrives:
  `upload 0.1–0.3 + STT 0.4–1.5 + rewrite 0–1.5 + embed 0.1–0.4 + retrieve 0.05–1 + generate 2–6 + TTS 1–3 + download 0.3–1.5`
  ≈ **5–12 s** (typical ~7 s). Generation is ~40% of it. TTS plus the payload are another ~25–30%.
* **Text turn, `/ask`.** `rewrite 0–1.5 + embed 0.1–0.4 + retrieve 0.05–1 + generate 2–6` ≈ **2.5–8 s**.
  The time to first visible text equals the total, because nothing streams.
* **Spoken turn with streaming (target).** `upload + STT + embed + retrieve + LLM TTFT (0.3–1 s) + first sentence (~15–25 tokens, 0.2–0.5 s) + one-sentence TTS (0.2–0.6 s) + one small chunk`
  ≈ **2–3.5 s**. The rest of the answer is synthesized while the first sentence plays.

The existing `Stage timings` log line (`pipeline.py:585-591`) covers only rewrite, embed, retrieve and generate.
It does not cover STT, auth, TTS, serialization or total time. `backend.log` had **0** such lines: no chat turn had run.

---

## 2. Measurements I ran

Benchmarks were throwaway scripts run with the project's virtualenv (Python 3.12, faiss 1.15.1, mongomock 4.3).
`src.globals` was stubbed so the import builds no DAOs. Whisper, Piper, embedding models and the network were not used.
The machine had ~200–400 MB free RAM, so the corpora are small.

### 2.1 Retrieval (`python bench.py r300 | r1000 | r4096`)
Synthetic chunks of 4000 chars (the production chunk size), 10 documents, top_k=5.

```
=== retrieval: 300 chunks x 4000 chars, dim=768 ===
BSON full docs: 4.2 MB on the wire, decode 13 ms
BSON w/o embedding (projection): 1.3 MB, decode 1 ms
_calculate_bm25_scores over 300 chunks: 77 ms
keyword_search() end-to-end on mongomock: 141 ms
FAISS IndexIDMap2(FlatIP) search k=25: 0.955 ms
5x find_one: 5 ms  vs  1x find($in): 3 ms (mongomock)

=== retrieval: 1000 chunks x 4000 chars, dim=768 ===
BSON full docs: 14.1 MB on the wire, decode 39 ms
BSON w/o embedding (projection): 4.2 MB, decode 2 ms
_calculate_bm25_scores over 1000 chunks: 258 ms
keyword_search() end-to-end on mongomock: 457 ms
FAISS IndexIDMap2(FlatIP) search k=25: 0.932 ms

=== retrieval: 150 chunks x 4000 chars, dim=4096 (Qwen3-Embedding-8B, the Idun default) ===
BSON full docs: 9.1 MB on the wire  (~61 kB per chunk; embedding is 93% of it)
BSON w/o embedding (projection): 0.6 MB
_calculate_bm25_scores over 150 chunks: 36 ms
```
What this shows:
* BM25 costs **~0.26 ms per chunk per turn**, in pure Python under the GIL.
* With Qwen 4096-d embeddings, the scan transfers **~60 MB per turn** at 1000 chunks.
* mongomock is slower than real pymongo, whose decoding is in C. Over a real socket, a projection
  saves 70–93% of the bytes, plus their transfer time.

### 2.2 FAISS index type (`python bench.py faiss`)
```
n=10000 dim=768 (31 MB): flat 0.79 ms, HNSW 0.100 ms (build 1.1s)
n=2500 dim=4096 (41 MB): flat 2.25 ms, HNSW 0.713 ms (build 2.7s)
```
FlatIP stays in the low milliseconds at the corpus sizes that are realistic per agent. Changing the index type gains nothing until ~10^5–10^6 vectors.

### 2.3 Prompt assembly and parsing (from the `r1000` run)
```
history=  2 turns: build_messages 0.17 ms, 24833 chars (~6208 tokens; reference block ~5218 tok)
history= 20 turns: build_messages 0.16 ms, 28498 chars (~7124 tokens)
history=100 turns: build_messages 0.17 ms, 44670 chars (~11167 tokens)
_parse_llm_response: valid 0.004 ms, truncated 0.047 ms; JSON scaffolding = 60 chars (~20 tokens) on a 424-char reply
```
Assembly and parsing cost nothing. Prompt *size* matters: 6–11k input tokens on every turn. The JSON contract costs ~20 output tokens, about 0.2–0.4 s.

### 2.4 Rewrite trigger rate (`python rewrite_trigger.py`)
```
6/20 standalone questions trigger the rewrite LLM call:
   Is it safe to touch the wire? -> ['it']
   Tell me about the history of this ship. -> ['this']
   Hva gjør den røde ventilen? -> ['den']
   Hva skal jeg gjøre her? -> ['her']
   Hvor lang tid tar det? -> ['det']
   Er det farlig å ta på ledningen? -> ['det']
```

### 2.5 Speech payload (`python bench.py tts`; stub synthesizer at 22.05 kHz, 16-bit mono; real split, concat, base64 and JSON code)
```
5 sentences, 30.5 s audio; split+concat+b64 overhead 18.6 ms; json.dumps 6.8 ms
raw PCM WAV 1.34 MB -> JSON body 3.59 MB (audio sent twice: speech + segments, base64 +33%)
opus @24 kbps would be ~91 kB for the same audio
  transfer at 20 Mbit/s: JSON 1435 ms vs single raw WAV 538 ms
  transfer at 50 Mbit/s: JSON 574 ms vs single raw WAV 215 ms
```

---

## 3. Ranked options

Score = (gain in s on a typical spoken turn × confidence) / effort. Effort: 0.5 = one line, 1 = S, 2 = M, 5 = L (backend and Unity).

| Rank | Option | Gain (typical) | Conf. | Effort | Score | Notes |
|---|---|---|---|---|---|---|
| 1 | **Shorter spoken answers.** Persona or system rule "1–3 sentences", and `llm_max_tokens` of ~200–300 for speech agents | 1–2 s (fewer output tokens, less TTS, smaller payload) | 0.6 | 0.5–1 | ~1.0 | A config and content decision. Truncated JSON is already recovered (`pipeline.py:226-251`). |
| 2 | **LLM streaming, sentence-chunked TTS and a streamed response (SSE or WebSocket)** | 3–5 s time to first audio; 1.5–4 s time to first text | 0.8 | 5 | ~0.65 | The largest absolute gain. See §4.1. |
| 3 | **Fix the rewrite** (tighter trigger, temperature 0, small model, or speculative parallel) | 0.5–1.5 s on the ~30%+ of turns where it fires | 0.6 | 1 | ~0.5 | See §4.2. |
| 4 | **Slim the speech payload** (send the audio once; later, binary or Opus) | 0.3–1 s (transfer and client decode) | 0.7 | 1–2 | ~0.35 | Send `speech_segments` *or* `speech`, chosen by a request flag. The web chat needs only `speech`. Opus would cut the bytes ~15×, but needs a Unity decoder. |
| 5 | **Keyword search: projection now, cached BM25 index later** | 0.05–1 s, growing with corpus | 0.9 | 1 (projection) / 2 (cache) | ~0.3 | See §4.3. Measured: 70–93% fewer bytes, and 258 ms of Python saved per 1000 chunks. |
| 6 | **Startup warmups** in `lifespan`: Whisper, Piper default voice, and one decode | 2–6 s on the **first** turn after a deploy or restart | 0.9 | 0.5 | (first turn only) | `main.py:47-50` has no warmup today. Costs RAM at boot. Make it an env flag. |
| 7 | **Smaller prompt**: history cap 100 → 10–20 turns, chunks 4000 → ~1000–1500 chars, or top_k 3 | 0.2–1 s TTFT (prefill), and cost | 0.5 | 1 (cap) / 2 (re-chunk and reindex) | ~0.2 | Prefill time depends heavily on provider; on Idun it is likely larger. Re-chunking also improves citation precision. |
| 8 | **Whisper settings**: pin the language when the client knows it, `vad_filter=true`, `without_timestamps=True`, `condition_on_previous_text=False`, explicit `cpu_threads`; client sends 16 kHz mono | 0.1–0.5 s | 0.6 | 1 | ~0.2 | Skips the language-detection pass and trims silence. Do not go above `base` on CPU. `tiny` is faster but worse for Norwegian. |
| 9 | **Idun keep-alive**: one `requests.Session` per `IdunLLM` | 30–150 ms per Idun call (×2 when rewriting) | 0.9 | 0.5 | ~0.15 | `llm.py:125`. The client is already cached by `create_llm`, so the session is reused. |
| 10 | **Cache of query embeddings** (LRU keyed by model and normalized text) | 0.1–0.4 s, but only on exact repeats | 0.9 | 0.5 | ~0.1 | Helps scripted and button prompts and greetings in XR scenes. |
| 11 | **Provider choice**: measure Idun against gpt-4o-mini or gemini-flash with the `generate` timing | Unknown, possibly 1–3 s | 0.4 | 0.5 (config) | n/a | Hosted flash and mini models usually have TTFT <1 s and 100+ tok/s. Idun depends on the model size and the shared-GPU queue. GDPR may decide this, not speed. |
| 12 | **Overlap pre-work with STT**: run auth, agent, keys and progress in parallel with Whisper | 10–30 ms | 0.8 | 1 | ~0.02 | Speculative retrieval cannot start before the transcript exists. Streaming STT (client-side VAD and chunked upload) could save 0.5–1 s, but at L effort. |
| 13 | **Batch the vector-hit lookups** (`find({"chunk_id": {"$in": ids}})`) and add an `(agent_id, chunk_id)` index | 2–10 ms | 0.9 | 0.5 | ~0.02 | `hybrid_search.py:374-383`. Also removes the per-agent scan behind each `find_one`. |
| 14 | **Drop the JSON citation contract** or use native structured output | ~0.2–0.4 s (20 tokens) | 0.5 | 3 | ~0.05 | Not worth it. "message" already comes first, so streaming can pull it out incrementally (§4.1). |
| 15 | HTTP keep-alive or HTTP/2 from Unity to nginx | 10–50 ms per turn | 0.5 | 1 | ~0.02 | UnityWebRequest mostly reuses connections already. Check that nginx keepalive is on. |
| 16 | Several uvicorn workers instead of the threadpool | ~0 for a single turn; large under concurrency | 0.5 | 2 | (throughput) | BM25, base64 and JSON hold the GIL. Caveats: Whisper, Piper and FAISS memory grow ×N; semaphores apply per process, so CPU limits multiply. FAISS reads across processes are OK (stat check `refresh()`, `faiss_vector_store.py:237-255`). Concurrent *uploads* to the same agent in two workers can lose updates, so keep ingestion on one worker or add a file lock. |
| 17 | FAISS HNSW or IVF | 0 at current scale | 0.9 | 2 | 0 | Flat search is under 1–2.3 ms (measured). Revisit above 10^5 vectors per agent. |

---

## 4. Top 3 in detail (engineering items; items 1 and 6 above are config quick wins to do now)

### 4.1 Streaming end to end: tokens, then sentences, then TTS, then audio chunks
**Change**
1. `src/llm.py`: add `generate_stream(messages, max_tokens, temperature) -> Iterator[str]` to each provider:
   * OpenAI: `chat.completions.create(..., stream=True)`.
   * Gemini: `models.generate_content_stream`.
   * Idun: `stream: true` SSE through a `requests.Session` with `stream=True` and `iter_lines`.
   Keep `generate()` for `/ask`, so its behaviour is unchanged.
2. `src/pipeline.py`: split `assemble_prompt_with_agent` (`:470-644`) into:
   * `prepare_turn()`: keys, rewrite, retrieve, `build_messages`.
   * `stream_turn()`: yields text deltas.
   * `finalize_turn()`: runs the existing `_parse_llm_response` on the full text and builds the existing response dict (citations, functions, grounded).
   Add an incremental extractor that emits only the decoded `"message"` string value while JSON streams. "message" is already the first key in both contracts (`:80-83`, `:375`). Reuse the `_CONTRACT_START_RE` idea. If the reply does not start with `{`, treat it as prose.
3. `src/tts.py`: an incremental sentence splitter over the delta stream that reuses `split_sentences` and its abbreviation rules. Submit each complete sentence to a `ThreadPoolExecutor`, keep the order, and yield `(index, text, wav_bytes)` as each finishes.
4. `src/routes/chat.py`: a new endpoint, `POST /api/chat/askTranscribeStream` (SSE via `StreamingResponse` with a sync generator, which FastAPI iterates in the threadpool), or a WebSocket. The events are:
   * `transcription`
   * `text` (deltas)
   * `audio` (index, sample rate, raw PCM or WAV base64 per sentence)
   * `final` (the current response dict with citations and `function_calls`)
   * `error`

   Keep the old endpoints for older Unity builds.
5. Unity: an SSE or WebSocket reader that queues `AudioClip`s per sentence and starts playback on the first one. The web chat can use `fetch` with `getReader()`.

**Risks**
* Function calls and `grounded` are known only at the end, so the client must not act on text before `final`.
* An HTTP error after streaming has started must arrive as an `error` event; the status code is already 200.
* The incremental JSON extractor must handle escapes. Fuzz-test it against `_recover_message`.
* nginx buffering: set `X-Accel-Buffering: no`, or `proxy_buffering off`, for this route.
* Sentence-level TTS slightly changes prosody across sentence boundaries. This is the same as today's per-sentence synthesis.
* Gemini and Idun stream formats differ, so each provider needs a contract test.

**Measure**
* Add server timestamps relative to request start: `t_transcribed`, `t_first_token`, `t_first_sentence`, `t_first_audio_sent`, `t_done`. Log them next to `Stage timings`.
* Primary KPI: **time to first audio** (`t_first_audio_sent`, and on the client, mic release until the first sample plays), compared with today's total `askTranscribeWithSpeech` latency on the same prompts and agent.
* Secondary KPI: time to first text on `/ask` versus the stream.

### 4.2 Make the rewrite cheap or remove it from the critical path
**Change** (`src/pipeline.py:112-148`, `:408-467`)
* (a) Tighten the trigger. Drop `her`, `de`, `det` and `den` when they are not sentence-initial, or require that the message is short (e.g. 8 tokens or fewer) *and* contains a pronoun. Better: rewrite only when the raw-query retrieval scores are weak (the top hybrid score is below the threshold plus a margin), which needs one extra retrieval.
* (b) When it does run, use `temperature=0` and an optional `agent.rewrite_model` or `REWRITE_LLM_MODEL` setting, for example a small Idun or flash model. Lower `REWRITE_MAX_TOKENS` to ~40.
* (c) Speculative option: start `embed(raw_query)` and retrieval in a thread *at the same time* as the rewrite call. Use the raw results if the rewrite fails or times out after a short budget (e.g. 600 ms). Otherwise embed the rewritten query. Wall time becomes `max(rewrite, embed)` plus one embed, instead of `rewrite + embed`.

**Risks**
* Retrieval quality on true follow-ups ("what about its weight?"). Check with a small labelled set of multi-turn questions: recall@k against the current behaviour.
* A different rewrite model means a second provider key.

**Measure**
* `timings["rewrite"]` (already logged), plus a new counter for how often the rewrite fired.
* Compare the p50 and p95 of `rewrite + embed + retrieve` before and after on replayed conversations. Compare citation hit rates.

### 4.3 Retrieval keyword path: stop streaming the corpus out of Mongo on every turn
**Change** (`src/rag_service/dao/context/hybrid_search.py:146-231`)
* Step 1, one line: `context_collection.find(query, {"embedding": 0})`. Measured: 70% fewer bytes at 768-d, 93% at 4096-d. Also replace the `next(...)` linear scan (`:206-218`) with a dict that maps `chunk_id` to its document. Fix the misleading docstring at `:157`.
* Step 2, a BM25 cache per agent: keep the tokenized corpus, document frequencies and average length in memory. Key it by `(agent_id, frozenset(available_documents))`, or keep one per agent and filter by document at scoring time. Invalidate it from `insert_context`, `delete_by_document_id` and `delete_chunks` (`mongodb_context_dao.py`), or with a cheap per-agent "chunk count and max `_id`" check.
  The FAISS store already keeps the chunk text in its metadata, so the BM25 cache could be built from the same place and share `refresh()` invalidation. This removes the per-turn Mongo scan entirely. Optionally use `rank_bm25` or a numpy sparse matrix for scoring.
* Step 3: add the `(agent_id, chunk_id)` index and batch the vector-hit lookups into a single `$in` (`:374-383`).

**Risks**
* Cache staleness after uploads, especially with multiple workers. Use the same disk or "version" signature pattern as `FAISSVectorStore.refresh()`.
* Memory: about the size of the text, i.e. 4 MB per 1000 chunks, which is acceptable.
* BM25 scores must stay identical. Add a unit test that compares the cached scores with `_calculate_bm25_scores`.

**Measure**
* `timings["retrieve"]` in `Stage timings`, at p50 and p95, on agents with 100, 1000 and 5000 chunks.
* Also watch process RSS, and Mongo network bytes (`db.serverStatus().network`) per turn.

---

## 5. What I could not measure
* **Real Whisper or Piper timings** on this hardware or in the container. They were not loaded because of the memory limit (only ~200–400 MB free during the spike). The STT and TTS figures above are typical numbers for faster-whisper `base` int8 and Piper `*-medium` on CPU, not measurements of this deployment.
* **LLM and embedding latency per provider** (Idun, OpenAI, Gemini), for both time to first token and tokens per second, and the cost of rewrite round trips. There were no API keys, and the running backend at `127.0.0.1:8000` had **no `Stage timings` lines** in `backend.log`: only admin and CRUD traffic had run.
* **Real Mongo RTT and transfer** for the keyword scan. mongomock measures Python CPU only. The byte counts come from BSON encoding of the same documents, so they are what would cross the wire.
* **Unity side**: JSON and base64 parse time on Quest, `UnityWebRequest` connection reuse, and how Unity uses `speech_segments` (lip sync?). The client code was not inspected; check this before dropping either audio field.
* **Concurrency behaviour**: several headsets at once, contention between threadpool and GIL (BM25 is pure Python), and STT and TTS semaphore queueing.
* **Suggested first step:** replay ~20 recorded turns against a staging stack with real keys. Extend the `Stage timings` line with `stt`, `tts`, `serialize` and `total`, so the estimates in §1 can be replaced with p50 and p95 figures.
