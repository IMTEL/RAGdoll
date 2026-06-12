# Local Translation Setup

RAGdoll uses LibreTranslate for GDPR-friendly translation.

The translation path is:

```text
RAGdollLanguage -> RAGdoll backend /api/translate -> LibreTranslate container
```

LibreTranslate is self-hosted and powered by open-source Argos Translate models. This means translation can run inside your own Docker environment instead of sending student text to Google Translate or another external API.

## Services

The Docker Compose files include a service:

```text
libretranslate
```

Production/internal URL:

```text
http://libretranslate:5000
```

Local development URL from the host:

```text
http://localhost:5000
```

## Environment Variables

RAGdoll backend:

```env
TRANSLATION_PROVIDER=libretranslate
TRANSLATION_BASE_URL=http://libretranslate:5000
TRANSLATION_API_KEY=
TRANSLATION_TIMEOUT_SECONDS=30
TRANSLATION_CACHE_TTL_SECONDS=3600
TRANSLATION_CACHE_MAX_ENTRIES=2000
```

LibreTranslate:

```env
LT_LOAD_ONLY=en,es
```

`LT_LOAD_ONLY` limits the downloaded/loaded models. The default is `en,es` because those packages are the required pair for the Spanish learning app and are known to work reliably with LibreTranslate.

Norwegian can be added later if the LibreTranslate image has a matching Argos package for the exact language code you need. Do not add unsupported language codes here, because LibreTranslate can boot with zero loaded languages and crash during startup.

## Start Locally

From `RAGdoll`:

```powershell
docker compose -f docker-compose.local.yml up -d --build
```

The first startup can take time because LibreTranslate downloads translation models into:

```text
ragdoll_libretranslate_data
```

The compose file starts LibreTranslate with:

```text
--load-only ${LT_LOAD_ONLY:-en,es} --update-models
```

This makes model installation explicit during container startup. The backend waits for LibreTranslate's `/languages` endpoint before it starts.

## Performance Behavior

Translation is intentionally on-demand in the language app. Agent responses are shown immediately in the target language, and English translation is requested only when the user clicks a word or the Translate button.

RAGdoll also keeps a short-lived in-memory translation cache:

```env
TRANSLATION_CACHE_TTL_SECONDS=3600
TRANSLATION_CACHE_MAX_ENTRIES=2000
```

This cache is per backend process and is not stored in MongoDB. That is deliberate: repeated words and phrases become fast, but student text is not retained permanently.

Set `TRANSLATION_CACHE_TTL_SECONDS=0` to disable backend translation caching.

## Test LibreTranslate Directly

From the host:

```powershell
Invoke-RestMethod -Method Post http://localhost:5000/translate `
  -ContentType "application/json" `
  -Body '{"q":"Hola, como estas?","source":"es","target":"en","format":"text"}'
```

Expected response:

```json
{
  "translatedText": "Hello, how are you?"
}
```

## Test Through RAGdoll

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/translate `
  -ContentType "application/json" `
  -Body '{"text":"Hola, como estas?","source":"es","target":"en"}'
```

Expected response shape:

```json
{
  "text": "Hola, como estas?",
  "translatedText": "Hello, how are you?",
  "source": "es",
  "target": "en",
  "provider": "libretranslate"
}
```

## Batch Translation

Use the batch endpoint when a client needs multiple translations at once:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/translate/batch `
  -ContentType "application/json" `
  -Body '{"texts":["hola","gracias"],"source":"es","target":"en"}'
```

Expected response shape:

```json
{
  "translations": [
    {
      "text": "hola",
      "translatedText": "hello",
      "source": "es",
      "target": "en",
      "provider": "libretranslate",
      "cached": false
    }
  ]
}
```

## Warm Translation

Call warmup after startup or when opening a translation feature:

```powershell
Invoke-RestMethod -Method Post "http://localhost:8000/api/translate/warmup?source=es&target=en"
```

The language app calls this automatically after connecting to an agent. The first manual translation may still be slower after a cold container start, but later requests should be faster.

## GDPR Notes

This setup is the preferred option for sensitive student data because translation requests stay inside the self-hosted Docker network.

Avoid `googletrans` for GDPR-sensitive use. It uses unofficial Google Translate web endpoints and does not provide the contractual/data-processing guarantees required for sensitive deployments.

External APIs can only be considered if the organization has a proper data processing agreement, approved data handling policy, and logging/retention controls.


## If LibreTranslate Crashes With Empty Languages

If the logs contain:

```text
IndexError: list index out of range
```

from `language_target_fallback`, LibreTranslate has started with no loaded languages. Check that `LT_LOAD_ONLY` only contains supported language codes. For this project, start with:

```env
LT_LOAD_ONLY=en,es
```

Then recreate the translation container:

```powershell
docker compose -f docker-compose.local.yml up -d libretranslate
```

If it still fails, the model volume may contain an incomplete download. Stop the service, remove only the LibreTranslate model volume, and start it again:

```powershell
docker compose -f docker-compose.local.yml stop libretranslate
docker volume rm ragdoll_libretranslate_data
docker compose -f docker-compose.local.yml up -d libretranslate
```

This removes downloaded translation models only. It does not remove MongoDB, Keycloak, agents, documents, access keys, Piper voices, or FAISS indexes.
