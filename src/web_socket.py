
# pip install webrtcvad

#  src/ws_stream.py
import asyncio, json, uuid, struct, tempfile
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import whisper
from src.transcribe import model as whisper_model           # re‑use loaded Whisper model
from src.pipeline import assemble_prompt                    # existing RAG → LLM helper
from src.LLM import create_llm                              # existing facade
import webrtcvad

router = APIRouter()
MAX_SECONDS = 15                     # hard‑stop per utterance
SAMPLE_RATE  = 16000

vad = webrtcvad.Vad(2)
FRAME_MS = 20
FRAME_BYTES = FRAME_MS * SAMPLE_RATE // 1000 * 2  

class ClientState:
    """Holds buffers + state for a single websocket client."""
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.audio = deque()          # deque[bytes]
        self.started = asyncio.Event()   # becomes true after first chunk
        self.closed  = False
        self.speech_active = True

    async def send_json(self, obj: dict):
        await self.ws.send_text(json.dumps(obj))

@router.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    state = ClientState(ws)

    # launch background task that monitors speech end & runs LLM
    llm_task = asyncio.create_task(handle_stream(state))

    try:
        while True:
            frame = await ws.receive_bytes()  # raw PCM16LE @16 kHz
            if not state.started.is_set():
                state.started.set()
            state.audio.append(frame)
            # keep only latest MAX_SECONDS for VAD/transcription windows
            if sum(len(b) for b in state.audio) > MAX_SECONDS * SAMPLE_RATE * 2:
                state.audio.popleft()
    except WebSocketDisconnect:
        state.closed = True
        await llm_task
        
        
        
# ================================================================
# Background task to handle LLM processing
# ================================================================



async def handle_stream(state: ClientState):
    partial_transcript = ""
    speech_decisions = deque(maxlen=30)  # 30 × 20 ms = 600 ms window
    llm_sent = False

    while not state.closed:
        await asyncio.sleep(0.02)  # 20 ms tick

        # Pull out any complete 20 ms frames and feed to VAD
        pcm_buf = b"".join(state.audio)
        while len(pcm_buf) >= FRAME_BYTES:
            frame = pcm_buf[:FRAME_BYTES]
            pcm_buf = pcm_buf[FRAME_BYTES:]
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            speech_decisions.append(is_speech)
        # store remainder back
        state.audio.clear()
        if pcm_buf:
            state.audio.append(pcm_buf)

        # Detect end‑of‑speech: majority = False for whole window
        if speech_decisions and not any(speech_decisions) and not llm_sent and partial_transcript:
            await state.send_json({"type": "final", "text": partial_transcript})
            # ... (LLM streaming unchanged) ...
            llm_sent = True
            continue

        # Every 400 ms run incremental transcription
        if not llm_sent and len(speech_decisions) % 20 == 0:  # ~400 ms
            if len(pcm_buf) < SAMPLE_RATE * 2:  # need ≥1 s audio to get decent partials
                continue
            wav16 = np.frombuffer(pcm_buf, np.int16).astype(np.float32) / 32768
            wav16 = whisper_model.pad_or_trim(wav16)
            mel = whisper_model.log_mel_spectrogram(wav16).to(whisper_model.device)
            dec = whisper_model.decode(mel, whisper.DecodingOptions(language="en"))
            if dec.text.strip():
                partial_transcript = dec.text.strip()
                await state.send_json({"type": "partial", "text": partial_transcript})