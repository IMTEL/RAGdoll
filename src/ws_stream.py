# src/ws_stream.py
import asyncio, json, uuid, struct, tempfile
from collections import deque
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import whisper
from src.transcribe import model as whisper_model
from src.pipeline import assemble_prompt
from src.LLM import create_llm
import webrtcvad

router = APIRouter()
MAX_SECONDS = 15
SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_BYTES = FRAME_MS * SAMPLE_RATE // 1000 * 2

class ClientState:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.audio = deque(maxlen=MAX_SECONDS * 1000 // FRAME_MS)  # Fixed size buffer
        self.started = asyncio.Event()
        self.closed = False
        self.speech_active = True

    async def send_json(self, obj: dict):
        try:
            await self.ws.send_text(json.dumps(obj))
        except:
            self.closed = True

@router.websocket("/ws/stream")
async def stream_endpoint(ws: WebSocket):
    await ws.accept()
    state = ClientState(ws)

    # Set a timeout for the connection
    try:
        # Main loop with timeout handling
        while not state.closed:
            try:
                frame = await asyncio.wait_for(
                    ws.receive_bytes(),
                    timeout=10.0  # Timeout if no data received for 10 seconds
                )
                if not state.started.is_set():
                    state.started.set()
                state.audio.append(frame)
                
            except asyncio.TimeoutError:
                state.closed = True
                break
            except WebSocketDisconnect:
                state.closed = True
                break
            except Exception:
                state.closed = True
                break
                
    finally:
        state.closed = True

async def handle_stream(state: ClientState):
    partial_transcript = ""
    speech_decisions = deque(maxlen=30)  # 600ms window
    llm_sent = False

    while not state.closed:
        try:
            await asyncio.sleep(0.02)  # 20ms tick

            # Process audio frames
            pcm_buf = b"".join(state.audio)
            while len(pcm_buf) >= FRAME_BYTES:
                frame = pcm_buf[:FRAME_BYTES]
                pcm_buf = pcm_buf[FRAME_BYTES:]
                is_speech = webrtcvad.Vad(2).is_speech(frame, SAMPLE_RATE)
                speech_decisions.append(is_speech)

            # Store remainder
            state.audio.clear()
            if pcm_buf:
                state.audio.append(pcm_buf)

            # Check for speech end
            if (speech_decisions and 
                not any(speech_decisions) and 
                not llm_sent and 
                partial_transcript):
                await state.send_json({"type": "final", "text": partial_transcript})
                llm_sent = True
                continue

            # Run transcription periodically
            if not llm_sent and len(speech_decisions) % 20 == 0:  # ~400ms
                if len(pcm_buf) < SAMPLE_RATE * 2:  # Need ≥1s audio
                    continue
                    
                wav16 = np.frombuffer(pcm_buf, np.int16).astype(np.float32) / 32768
                wav16 = whisper.pad_or_trim(wav16)
                mel = whisper.log_mel_spectrogram(wav16).to(whisper_model.device)
                dec = whisper.decode(mel, whisper.DecodingOptions(language="en"))
                
                if dec.text.strip():
                    partial_transcript = dec.text.strip()
                    await state.send_json({"type": "partial", "text": partial_transcript})

        except Exception as e:
            state.closed = True
            break