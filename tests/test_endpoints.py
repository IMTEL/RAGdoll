# test_app.py

import json
import io
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI, File, UploadFile, Form

from src.main import app

client = TestClient(app)

# Use a small dummy WAV header (not playable, but passes the file check)
DUMMY_WAV = (
    b"RIFF$\x00\x00\x00WAVEfmt " +
    b"\x10\x00\x00\x00\x01\x00\x01\x00" +
    b"\x40\x1f\x00\x00\x80>\x00\x00" +
    b"\x02\x00\x10\x00data\x00\x00\x00\x00"
)

# Sample minimal JSON payload
TEST_JSON = """{
    "action": "test",
    "value": 42
}"""

@pytest.fixture
def dummy_audio_file():
    return io.BytesIO(DUMMY_WAV)

def test_ask_transcribe_with_mock_audio(dummy_audio_file):
    response = client.post(
        "/askTranscribe",
        files={"audio": ("test.wav", dummy_audio_file, "audio/wav")},
        data={"data": (TEST_JSON)}
    )
    assert response.status_code == 200
    # assert "response" in response.json()

def test_ask_transcribe_invalid_json(dummy_audio_file):
    response = client.post(
        "/askTranscribe",
        files={"audio": ("test.wav", dummy_audio_file, "audio/wav")},
        data={"data": "not a json"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Invalid command."
