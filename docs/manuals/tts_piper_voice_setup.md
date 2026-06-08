# TTS Piper Voice Setup

RAGdoll text-to-speech uses Piper locally. The backend does not call a cloud TTS provider, but it needs Piper voice model files installed on the server.

The backend expects voice files in:

```text
/data/piper/voices
```

In Docker, this path is backed by the volume:

```text
ragdoll_piper_voices
```

Each voice needs two files:

```text
VOICE_NAME.onnx
VOICE_NAME.onnx.json
```

## Configured Voices

The current default voices are:

| Language | Env var | Voice file prefix |
| --- | --- | --- |
| English | `TTS_DEFAULT_VOICE_EN` | `en_US-lessac-medium` |
| Norwegian | `TTS_DEFAULT_VOICE_NO` | `no_NO-talesyntese-medium` |
| Spanish | `TTS_DEFAULT_VOICE_ES` | `es_ES-davefx-medium` |

These values are configured in `RAGdoll/.env`:

```env
TTS_ENGINE=piper
TTS_VOICE_DIR=/data/piper/voices
TTS_DEFAULT_LANGUAGE=en
TTS_DEFAULT_VOICE_EN=en_US-lessac-medium
TTS_DEFAULT_VOICE_NO=no_NO-talesyntese-medium
TTS_DEFAULT_VOICE_ES=es_ES-davefx-medium
TTS_USE_CUDA=false
TTS_WARMUP_TEXT=Ready.
```

## Install Voices on the Server

Run these commands from the server. They download the voice files into the Docker volume used by the backend.

Create the volume if it does not already exist:

```bash
sudo docker volume create ragdoll_piper_voices
```

Download English:

```bash
sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx

sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Download Norwegian:

```bash
sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/no_NO-talesyntese-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx

sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/no_NO-talesyntese-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/no/no_NO/talesyntese/medium/no_NO-talesyntese-medium.onnx.json
```

Download Spanish:

```bash
sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/es_ES-davefx-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx

sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  curlimages/curl:8.11.1 \
  --fail --location \
  --output /voices/es_ES-davefx-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json
```

## Verify the Files

List the files in the volume:

```bash
sudo docker run --rm --user 0:0 \
  -v ragdoll_piper_voices:/voices \
  alpine:3.20 \
  ls -lh /voices
```

Expected files:

```text
en_US-lessac-medium.onnx
en_US-lessac-medium.onnx.json
no_NO-talesyntese-medium.onnx
no_NO-talesyntese-medium.onnx.json
es_ES-davefx-medium.onnx
es_ES-davefx-medium.onnx.json
```

If the backend container is running, verify that it sees the same files:

```bash
sudo docker exec ragdoll-backend sh -c "ls -lh /data/piper/voices"
```

## Restart or Rebuild the Backend

If the backend was already running before the `piper_voices` volume was added to `docker-compose.yml`, restart the stack:

```bash
sudo docker compose down
sudo docker compose up -d --build
```

If the server uses the older Docker Compose command:

```bash
sudo docker-compose down
sudo docker-compose up -d --build
```

## Test TTS Warmup

Test English:

```bash
sudo docker exec ragdoll-backend python -c "import requests; r=requests.post('http://localhost:8000/api/chat/tts/warmup', json={'language':'en'}, timeout=60); print(r.status_code); print(r.text)"
```

Test Norwegian:

```bash
sudo docker exec ragdoll-backend python -c "import requests; r=requests.post('http://localhost:8000/api/chat/tts/warmup', json={'language':'no'}, timeout=60); print(r.status_code); print(r.text)"
```

Test Spanish:

```bash
sudo docker exec ragdoll-backend python -c "import requests; r=requests.post('http://localhost:8000/api/chat/tts/warmup', json={'language':'es'}, timeout=60); print(r.status_code); print(r.text)"
```

Expected result:

```text
200
{"success":true,...}
```

## Local Windows Shortcut

For local Docker testing on Windows, use the included PowerShell helper:

```powershell
.\scripts\install-piper-voices.ps1
```

Install only one language:

```powershell
.\scripts\install-piper-voices.ps1 -Voices en
```

## Troubleshooting

### `Piper voice was not found`

The backend cannot find the configured `.onnx` file in `/data/piper/voices`.

Check:

- The `ragdoll_piper_voices` volume exists.
- The backend service mounts `piper_voices:/data/piper/voices`.
- The `.onnx` and `.onnx.json` files both exist.
- The file prefix matches the `.env` value exactly.

### `Piper TTS is not installed`

The backend image was built before `piper-tts` was added. Rebuild:

```bash
sudo docker compose up -d --build backend-service
```

For local compose:

```bash
docker compose -f docker-compose.local.yml up -d --build backend
```

### Permission Errors During Download

Use `--user 0:0` in the temporary download container. Without it, `curlimages/curl` may not have permission to write into the Docker volume.
