# tests/test_web_socket.py
import time, types, numpy as np, pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.ws_stream import router as ws_router, SAMPLE_RATE, FRAME_BYTES
import whisper

# ---------- helpers ----------
def pcm_sine(ms=20, freq=440):
    t = np.linspace(0, ms / 1000, int(SAMPLE_RATE * ms / 1000), endpoint=False)
    return np.int16(0.1 * np.sin(2 * np.pi * freq * t) * 32768).tobytes()

SPEECH = pcm_sine()
SILENCE = b"\x00" * FRAME_BYTES

# ---------- monkey-patch Whisper ----------
@pytest.fixture(autouse=True)
def patch_whisper(monkeypatch):
    monkeypatch.setattr(whisper, "pad_or_trim", lambda x: x)
    monkeypatch.setattr(whisper, "log_mel_spectrogram", lambda x: x)
    monkeypatch.setattr(whisper, "decode",
                      lambda m, mel, opts: types.SimpleNamespace(text="hello world"))

# ---------- test client ----------
@pytest.fixture(scope="module")
def client():
    app = FastAPI()
    app.include_router(ws_router)
    return TestClient(app)

def test_websocket_connection(client):
    """Test basic WebSocket connection"""
    with client.websocket_connect("/ws/stream") as ws:
        # Just test connection works
        ws.send_bytes(SPEECH)
        response = ws.receive_json()
        assert "type" in response
        assert "text" in response

def test_partial_and_final(client):
    """Test the full speech/silence sequence"""
    with client.websocket_connect("/ws/stream") as ws:
        # Send speech (1.2 seconds)
        for _ in range(60):
            ws.send_bytes(SPEECH)
        
        # Wait for partial transcript
        partial = None
        for _ in range(10):  # Try up to 10 times
            try:
                msg = ws.receive_json(timeout=0.5)
                if msg["type"] == "partial":
                    partial = msg["text"]
                    break
            except:
                continue
        
        # Send silence (0.6 seconds)
        for _ in range(30):
            ws.send_bytes(SILENCE)
        
        # Wait for final transcript
        final = None
        for _ in range(10):  # Try up to 10 times
            try:
                msg = ws.receive_json(timeout=0.5)
                if msg["type"] == "final":
                    final = msg["text"]
                    break
            except:
                continue
        
        assert partial == "hello world"
        assert final == "hello world"