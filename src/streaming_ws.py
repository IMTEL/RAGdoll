"""streaming_ws.py
-----------------
Adds full‑duplex WebSocket streaming to the chat‑service for:
    •  client → server   : raw PCM audio (+ control JSON frames)
    •  server → client   : partial transcripts, final transcript, token‑stream from the LLM, (future) TTS audio.

The module is 100 % self‑contained: simply `import` and `include_router(router)` in **src/main.py** and rebuild
Docker to enable streaming.

Protocol (all messages share the same WebSocket):
─────────────────────────────────────────────────
bytes      – 16 kHz, 16‑bit little‑endian mono PCM audio
text/JSON  – {"type": "<event>", ...}
    client→server events
        • silence            – VR client detected end‑of‑utterance
        • command            – Full serialized `Command` object for RAG/LLM
    server→client events
        • transcript_partial – incremental Whisper output
        • transcript_final   – final Whisper output (after `silence`)
        • llm_token          – single token from the LLM stream
        • error              – unexpected problems
        • tts_chunk          – *(future)* base64/wav audio matching the `llm_token`

Unity clients can keep a single socket open per NPC and multiplex speech turns with
simple counters inside the JSON payload.
"""

import asyncio
import json
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.command import command_from_json, Command
from src.pipeline import assemble_prompt
from src.config import Config
from src.LLM import create_llm

# ──────────────────────────────────────────────────────────────────────────
# Whisper streaming helper (very light‑weight, 1–2 s latency)
# ──────────────────────────────────────────────────────────────────────────
import whisper

_MODEL = Config().whisper_model     # already GPU‑aware in existing code
_SAMPLE_RATE = 16_000               # Unity should down‑sample if needed
_CHUNK_S    = 1.0                   # seconds of audio per decode step
_CHUNK_BYTES = int(_SAMPLE_RATE * _CHUNK_S) * 2  # 16‑bit = 2 bytes

class WhisperStreamer:
    """Feed raw PCM chunks; yields partial transcripts every *_CHUNK_S* seconds."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, pcm_bytes: bytes) -> Optional[str]:
        self._buf.extend(pcm_bytes)
        if len(self._buf) < _CHUNK_BYTES:
            return None
        # Take one slice; leave remainder for next round
        slice_bytes = self._buf[:_CHUNK_BYTES]
        del self._buf[:_CHUNK_BYTES]
        audio_np = np.frombuffer(slice_bytes, np.int16).astype(np.float32) / 32768.0
        mel = whisper.log_mel_spectrogram(audio_np).to(_MODEL.device)
        opts = whisper.DecodingOptions(language="no", without_timestamps=True)
        dec  = whisper.decode(_MODEL, mel, opts)
        return dec.text.strip()

# ──────────────────────────────────────────────────────────────────────────
# LLM streaming helper (OpenAI today; easy to swap via factory)
# ──────────────────────────────────────────────────────────────────────────
# streaming_ws.py
async def stream_chat_completion(prompt: str, model: str = "openai") -> AsyncGenerator[str, None]:
    llm = create_llm(model)
    if hasattr(llm, "astream"):
        async for tok in llm.astream(prompt):
            yield tok
    else:                               # never hits OpenAI / Gemini after step 3
        yield llm.generate(prompt)


# ──────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

manager = ConnectionManager()
router  = APIRouter()

@router.websocket("/ws/chat")
async def chat_stream(websocket: WebSocket):
    """Main bidirectional stream used by Unity clients."""

    await manager.connect(websocket)
    transcriber = WhisperStreamer()
    final_transcript: str = ""
    pending_command_json: Optional[str] = None

    try:
        while True:
            message = await websocket.receive()

            # 1️  raw audio
            if message["type"] == "websocket.receive" and "bytes" in message:
                partial = transcriber.feed(message["bytes"] or b"")
                if partial:
                    await websocket.send_text(json.dumps({
                        "type": "transcript_partial",
                        "data": partial,
                    }))
                    final_transcript = partial  # keep updating – last one is full text

            # 2️ control / JSON frames
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error", "data": "Malformed JSON"}))
                    continue

                if data.get("type") == "silence":
                    # ─ End‑of‑utterance: flush any remaining audio immediately ─
                    if transcriber._buf:
                        partial = transcriber.feed(b"")
                        if partial:
                            final_transcript = partial
                    await websocket.send_text(json.dumps({
                        "type": "transcript_final",
                        "data": final_transcript,
                    }))

                    # Kick off RAG/LLM in background so that we can stream tokens
                    if pending_command_json is not None:
                        asyncio.create_task(_handle_llm(websocket, final_transcript, pending_command_json))
                        pending_command_json = None

                elif data.get("type") == "command":
                    # Full serialized Command arriving from Unity (one per turn)
                    pending_command_json = data["data"]

            # ─ any other message type is ignored ─

    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def _handle_llm(websocket: WebSocket, transcript: str, command_json: str):
    """Runs RAG + LLM, pushes token stream back to client."""
    try:
        command: Command = command_from_json(command_json, question=transcript)
        prompt_obj     = assemble_prompt(command)  # returns dict with .response key
        async for tok in stream_chat_completion(prompt_obj["response"], model="openai"):
            await websocket.send_text(json.dumps({
                "type": "llm_token",
                "data": tok,
            }))
    except Exception as exc:  # pragma: no cover – best‑effort error path
        await websocket.send_text(json.dumps({"type": "error", "data": str(exc)}))